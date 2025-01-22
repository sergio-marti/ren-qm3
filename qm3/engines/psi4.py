import  numpy
import  typing
import  os
import  glob
import  qm3.data
import  qm3.engines
import  psi4

#
# Set environment variable: PSI_SCRATCH
#

class run( qm3.engines.template ):
    def __init__( self, mol: object, 
            opts: typing.Optional[dict] = { "reference": "rks", "basis": "def2-svp", "d_convergence": 6, "scf_type": "direct",
                "guess": "read", "output": False, "charge": 0, "method": "b3lyp", "ncpus": 2, "memory": "4096 MB" },
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.cx  = 1.0 / qm3.data.A0
        self.ce  = qm3.data.H2J
        self.cg  = self.ce * self.cx

        buf = "\n%d 1\n"%( opts.pop( "charge" ) )
        for i in self.sel:
            buf += "%-2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        self.vla = []
        self.__dq = numpy.zeros( mol.natm )
        if( len( self.lnk ) > 0 ):
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                buf += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            # redistribute MM-charge on the remaining atoms of the group
            for i,j in self.lnk:
                if( j in self.grp ):
                    self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
            # ----------------------------------------------------------
        if( opts.pop( "output" ) ):
            psi4.core.set_output_file( "psi4.out", False )
        else:
            psi4.core.be_quiet()
        psi4.set_memory( opts.pop( "memory" ) )
        psi4.set_num_threads( opts.pop( "ncpus" ) )
        buf += "symmetry c1\nnoreorient\nnocom\nunits ang\n"
        self.aQM = psi4.geometry( buf )
        self.met = opts.pop( "method" )
        psi4.set_options( opts )


    def update_coor( self, mol ):
        if( self.img ):
            cen = numpy.mean( mol.coor[self.sel], axis = 0 )
        crd = []
        for i in self.sel:
            crd.append( ( self.cx * mol.coor[i] ).tolist() )
        self.vla = []
        if( len( self.lnk ) > 0 ):
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                crd.append( ( self.cx * c ).tolist() )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        self.aQM.set_geometry( psi4.core.Matrix.from_list( crd ) )
        self.aQM.update_geometry()
        self.aMM = None
        if( len( self.nbn ) > 0 ):
            self.aMM = []
            f = open( "grid.dat", "wt" )
            for i in self.nbn:
                tmp = mol.coor[i].copy()
                if( self.img ):
                    tmp -= mol.boxl * numpy.round( ( tmp - cen ) / mol.boxl, 0 )
                self.aMM.append( [ mol.chrg[i] + self.__dq[i], tmp[0] * self.cx, tmp[1] * self.cx, tmp[2] * self.cx ] )
                f.write( "%20.10lf%20.10lf%20.10lf\n"%( tmp[0], tmp[1], tmp[2] ) )
            f.close()
            self.aMM = numpy.array( self.aMM )


    def get_func( self, mol ):
        self.update_coor( mol )
        mol.func += psi4.energy( self.met, return_wfn = False, external_potentials = self.aMM ) * self.ce


    def get_grad( self, mol ):
        self.update_coor( mol )
        g, wfn = psi4.gradient( self.met, return_wfn = True, external_potentials = self.aMM )
        mol.func += psi4.variable( "CURRENT ENERGY" ) * self.ce
        g = ( self.cg * g.to_array() ).ravel()
        qm3.engines.Link_grad( self.vla, g )
        k = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[k]
                k += 1
        if( len( self.nbn ) > 0 ):
            ef = psi4.core.OEProp( wfn )
            ef.add( "GRID_FIELD" )
            ef.compute()
            efx = ef.Exvals()
            efy = ef.Eyvals()
            efz = ef.Ezvals()
            k = 0
            for i in self.nbn:
                mol.grad[i,0] -= self.cg * ( mol.chrg[i] + self.__dq[i] ) * efx[k]
                mol.grad[i,1] -= self.cg * ( mol.chrg[i] + self.__dq[i] ) * efy[k]
                mol.grad[i,2] -= self.cg * ( mol.chrg[i] + self.__dq[i] ) * efz[k]
                k += 1
