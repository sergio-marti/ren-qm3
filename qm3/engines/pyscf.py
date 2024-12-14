import  numpy
import  typing
import  qm3.data
import  qm3.engines

import  os
import  pyscf.gto
import  pyscf.dft
import  pyscf.qmmm
import  pyscf.grad
import  pyscf.lib
    

class run( qm3.engines.template ):
    def __init__( self, mol: object, 
            opts: typing.Optional[dict] = { "basis": "def2-svp", "conv_tol": 1.e-9, "charge": 0, "spin": 0,
                "method": "b3lypg", "memory": 4096, "grid": 3, "max_cyc": 50, "nproc": 2 },
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.cx  = 1.0 / qm3.data.A0
        self.ce  = qm3.data.H2J
        self.cg  = self.ce * self.cx
        self.opt = opts
        # redistribute MM-charge on the remaining atoms of the group
        self.__dq = numpy.zeros( mol.natm )
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------


    def update_coor( self, mol ):
        # just updating coordinates seems not to work properly...
        aQM = pyscf.gto.Mole()
        aQM.unit = "Angstrom"
        aQM.symmetry = False
        aQM.basis = self.opt["basis"]
        aQM.spin = self.opt["spin"]
        aQM.charge = self.opt["charge"]
        aQM.verbose = 0
        aQM.atom = ""
        for i in self.sel:
            aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        self.vla = []
        if( len( self.lnk ) > 0 ):
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        aQM.build()
        if( aQM.spin == 0 ):
            dft = pyscf.dft.RKS( aQM )
        else:
            dft = pyscf.dft.UKS( aQM )
        dft.verbose = 0
        dft.direct_scf = True
        dft.conv_tol = self.opt["conv_tol"]
        dft.max_cycle = self.opt["max_cyc"]
        dft.grids.level = self.opt["grid"]
        dft.xc = self.opt["method"]
        dft.max_memory = self.opt["memory"]
        dft.chkfile = "pyscf.chk"
        if( os.path.isfile( "pyscf.chk" ) ):
            dft.init_guess = "chkfile"
        if( len( self.nbn ) > 0 ):
            crd = mol.coor[self.nbn].copy()
            if( self.img ):
                for i in range( crd.shape[0] ):
                    crd[i] -= mol.boxl * numpy.round( crd[i] / mol.boxl, 0 )
            crd *= self.cx
            chg = mol.chrg[self.nbn] + self.__dq[self.nbn]
            scf = pyscf.qmmm.mm_charge( dft, crd, chg, unit = "Bohr" )
        else:
            scf = dft
        pyscf.lib.num_threads( self.opt["nproc"] )
        return( scf )


    def get_func( self, mol ):
        scf = self.update_coor( mol )
        out = scf.kernel() * self.ce
        mol.func += out
        chg = scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        return( out )


    def get_grad( self, mol ):
        scf = self.update_coor( mol )
        out = scf.kernel() * self.ce
        mol.func += out
        chg = scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        g = scf.Gradients().run( grid_response = True ).de.flatten().tolist()
        qm3.engines.Link_grad( self.vla, g )
        k = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[k] * self.cg
                k += 1
        if( len( self.nbn ) > 0 ):
            den = scf.make_rdm1()
            dr  = scf.mol.atom_coords()[:,None,:] - scf.mm_mol.atom_coords()
            r   = numpy.linalg.norm( dr, axis = 2 )
            g   = numpy.einsum( "r,R,rRx,rR->Rx", scf.mol.atom_charges(), scf.mm_mol.atom_charges(), dr, r ** -3 )
            if( len( den.shape ) == 3 ):
                for i,q in enumerate( scf.mm_mol.atom_charges() ):
                    with scf.mol.with_rinv_origin( scf.mm_mol.atom_coord( i ) ):
                        v = scf.mol.intor( "int1e_iprinv" )
                    g[i] += ( numpy.einsum( "ij,xji->x", den[0], v ) + numpy.einsum( "ij,xij->x", den[0], v.conj() ) ) * -q
                    g[i] += ( numpy.einsum( "ij,xji->x", den[1], v ) + numpy.einsum( "ij,xij->x", den[1], v.conj() ) ) * -q
            else:
                for i,q in enumerate( scf.mm_mol.atom_charges() ):
                    with scf.mol.with_rinv_origin( scf.mm_mol.atom_coord( i ) ):
                        v = scf.mol.intor( "int1e_iprinv" )
                    g[i] += ( numpy.einsum( "ij,xji->x", den, v ) + numpy.einsum( "ij,xij->x", den, v.conj() ) ) * -q
            k = 0
            for i in self.nbn:
                mol.grad[i,:] += g[k,:] * self.cg
                k += 1
        return( out )
