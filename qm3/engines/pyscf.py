import  numpy
import  typing
import  qm3.data
import  qm3.engines

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

        aQM = pyscf.gto.Mole()
        aQM.unit = "Angstrom"
        aQM.symmetry = False
        aQM.basis = opts.pop( "basis" )
        aQM.spin = opts.pop( "spin" )
        aQM.charge = opts.pop( "charge" )
        aQM.verbose = 0
        aQM.atom = ""
        for i in self.sel:
            aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        self.vla = []
        dq = numpy.zeros( mol.natm )
        if( len( self.lnk ) > 0 ):
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            # redistribute MM-charge on the remaining atoms of the group
            for i,j in self.lnk:
                if( j in self.grp ):
                    dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
            # ----------------------------------------------------------
        aQM.build()
        if( aQM.spin == 0 ):
            self.dft = pyscf.dft.RKS( aQM )
        else:
            self.dft = pyscf.dft.UKS( aQM )
        self.dft.verbose = 0
        self.dft.direct_scf = True
        self.dft.conv_tol = opts.pop( "conv_tol" )
        self.dft.max_cycle = opts.pop( "max_cyc" )
        self.dft.grids.level = opts.pop( "grid" )
        self.dft.xc = opts.pop( "method" )
        self.dft.max_memory = int( opts.pop( "memory" ) )
        if( len( self.nbn ) > 0 ):
            crd = mol.coor[self.nbn].copy()
            if( self.img ):
                for i in range( crd.shape[0] ):
                    crd[i] -= mol.boxl * numpy.round( crd[i] / mol.boxl, 0 )
            crd *= self.cx
            chg = mol.chrg[self.nbn] + dq[self.nbn]
            self.scf = pyscf.qmmm.mm_charge( self.dft, crd, chg, unit = "Bohr" )
        else:
            self.scf = self.dft
        pyscf.lib.num_threads( opts.pop( "nproc" ) )


    def update_coor( self, mol ):
        crd = []
        for i in self.sel:
            crd.append( mol.coor[i].tolist() )
        self.vla = []
        if( len( self.lnk ) > 0 ):
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                crd.append( c.tolist() )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        self.scf.mol.set_geom_( numpy.array( crd ) )
        if( len( self.nbn ) > 0 ):
            crd = mol.coor[self.nbn].copy()
            if( self.img ):
                for i in range( crd.shape[0] ):
                    crd[i] -= mol.boxl * numpy.round( crd[i] / mol.boxl, 0 )
            crd *= self.cx
            self.scf.mm_mol.set_geom_( crd )


    def get_func( self, mol ):
        self.update_coor( mol )
        out = self.scf.kernel() * self.ce
        mol.func += out
        chg = self.scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        return( out )


    def get_grad( self, mol ):
        self.update_coor( mol )
        out = self.scf.kernel() * self.ce
        mol.func += out
        chg = self.scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        g = self.scf.Gradients().run( grid_response = True ).de.flatten().tolist()
        qm3.engines.Link_grad( self.vla, g )
        k = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[k] * self.cg
                k += 1
        if( len( self.nbn ) > 0 ):
            den = self.scf.make_rdm1()
            dr  = self.scf.mol.atom_coords()[:,None,:] - self.scf.mm_mol.atom_coords()
            r   = numpy.linalg.norm( dr, axis = 2 )
            g   = numpy.einsum( "r,R,rRx,rR->Rx", self.scf.mol.atom_charges(), self.scf.mm_mol.atom_charges(), dr, r ** -3 )
            if( len( den.shape ) == 3 ):
                for i,q in enumerate( self.scf.mm_mol.atom_charges() ):
                    with self.scf.mol.with_rinv_origin( self.scf.mm_mol.atom_coord( i ) ):
                        v = self.scf.mol.intor( "int1e_iprinv" )
                    g[i] += ( numpy.einsum( "ij,xji->x", den[0], v ) + numpy.einsum( "ij,xij->x", den[0], v.conj() ) ) * -q
                    g[i] += ( numpy.einsum( "ij,xji->x", den[1], v ) + numpy.einsum( "ij,xij->x", den[1], v.conj() ) ) * -q
            else:
                for i,q in enumerate( self.scf.mm_mol.atom_charges() ):
                    with self.scf.mol.with_rinv_origin( self.scf.mm_mol.atom_coord( i ) ):
                        v = self.scf.mol.intor( "int1e_iprinv" )
                    g[i] += ( numpy.einsum( "ij,xji->x", den, v ) + numpy.einsum( "ij,xij->x", den, v.conj() ) ) * -q
            k = 0
            for i in self.nbn:
                mol.grad[i,:] += g[k,:] * self.cg
                k += 1
        return( out )
