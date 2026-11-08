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

_pbc = False
try:
    import  pyscf.qmmm.pbc.itrf
    _pbc = True
except Exception:
    pass

_gpu = False
try:
    import  gpu4pyscf.dft
    import  gpu4pyscf.qmmm.itrf
    import  gpu4pyscf.qmmm.pbc.itrf
    import  cupy
    cupy.cuda.runtime.getDeviceCount()
    _gpu = True
except Exception:
    pass


class run( qm3.engines.template ):
    """
    options:
        basis, method, charge, spin, conv_tol, max_cyc, grid, memory (MB), nproc (CPU),
        df (bool, auxiliary basis for density fitting),
        gpu (bool, run on GPU through gpu4pyscf),
        gpuid (int, GPU device ordinal),
        chk (bool or str, density matrix kept between calls and stored on file, default "pyscf_dm.npz"),
        pbc (bool, periodic boundary conditions via Ewald summation instead of
             minimum image convention; requires a valid molecular box),
        rcut_ewald (float, real-space Ewald cutoff in Angstrom, default half of the smallest box edge),
        rcut_hcore (float, cutoff in Angstrom for the exact QM-MM electrostatic coupling,
                    it has to enclose the whole QM region, default half of the box diagonal),
        radii (float, Gaussian spreading radius in Angstrom for the MM charges, default None,
               point charges; mandatory on GPU-PBC, where 0.1 Ang is set by default),
        grid_response (bool, DFT grid response term in the gradient, default True),
        guess (str, pyscf initial guess override, e.g. "hcore"; the stored density takes precedence)
    """
    def __init__( self, mol: object,
            opts: typing.Optional[dict] = { "basis": "def2-svp", "conv_tol": 1.e-9, "charge": 0, "spin": 0,
                "method": "b3lypg", "memory": 4096, "grid": 3, "max_cyc": 50, "nproc": 2, "df": True, "chk": False },
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.cx  = 1.0 / qm3.data.A0
        self.ce  = qm3.data.H2J
        self.cg  = self.ce * self.cx
        self.opt = opts
        self.gpu = bool( self.opt.get( "gpu", False ) )
        if( self.gpu and not _gpu ):
            raise RuntimeError( "gpu4pyscf/cupy are not available, unable to run on GPU" )
        self.pbc = bool( self.opt.get( "pbc", False ) )
        self.dev = int( self.opt.get( "gpuid", 0 ) )
        if( not _pbc ):
            self.pbc = False
            print( ">> pyscf.qmmm.pbc is unavailable, PBC embedding is disabled" )
        if( self.pbc ):
            if( numpy.max( mol.boxl ) > 1.0e10 ):
                raise RuntimeError( "invalid box dimensions for PBC embedding" )
            if( self.img ):
                print( ">> minimum image NBnd disabled, Ewald summation is used instead" )
                self.img_clean()
            if( self.opt.get( "radii", None ) is None and self.gpu ):
                self.opt = dict( self.opt )
                self.opt["radii"] = 0.1
                print( ">> point charges are not supported on GPU/PBC, 0.1 Ang Gaussian charges are used instead" )
        self.rad = self.opt.get( "radii", None )
        if( self.rad is not None and numpy.isscalar( self.rad ) ):
            self.rad = numpy.full( self.nbn.shape[0], float( self.rad ) )
        if( self.rad is not None and not self.pbc and self.gpu ):
            raise RuntimeError( "Gaussian MM charges are only supported in PBC mode on GPU" )
        self.chk = None
        if( self.opt.get( "chk", False ) ):
            self.chk = self.opt["chk"] if isinstance( self.opt["chk"], str ) else "pyscf_dm.npz"
        self.__dm = None
        # redistribute MM-charge on the remaining atoms of the group
        self.__dq = numpy.zeros( mol.natm )
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------
        print( ">> pyscf engine: %d QM + %d MM atoms, GPU:%s PBC:%s"%
            ( len( self.sel ) + len( self.lnk ), len( self.nbn ), self.gpu, self.pbc ) )


    def __chk_save( self ):
        if( not self.chk or self.__dm is None ):
            return
        try:
            tmp = "%s.%d.tmp"%( self.chk, os.getpid() )
            with open( tmp, "wb" ) as f:
                numpy.savez( f, dm = self.__dm )
            os.replace( tmp, self.chk )
        except Exception:
            try:
                os.remove( tmp )
            except Exception:
                pass


    def __chk_load( self, aqm ):
        if( not self.chk or not os.path.isfile( self.chk ) ):
            return None
        try:
            with numpy.load( self.chk, allow_pickle = False ) as dat:
                if( "dm" not in dat ):
                    return None
                dm = dat["dm"]
                if( dm.shape[-1] != aqm.nao ):
                    return None
                if( aqm.spin == 0 and dm.ndim != 2 ):
                    return None
                if( aqm.spin != 0 and ( dm.ndim != 3 or dm.shape[0] != 2 ) ):
                    return None
                return dm
        except Exception:
            return None


    def update_coor( self, mol ):
        if( self.gpu ):
            cupy.cuda.Device( self.dev ).use()
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
                c, v = qm3.engines.link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, self.lnk[i][1], v ) )
                k += 1
        aQM.build()
        if( self.gpu ):
            dft = gpu4pyscf.dft.RKS( aQM ) if( aQM.spin == 0 ) else gpu4pyscf.dft.UKS( aQM )
        else:
            dft = pyscf.dft.RKS( aQM ) if( aQM.spin == 0 ) else pyscf.dft.UKS( aQM )
        dft.verbose = 0
        dft.conv_tol = self.opt["conv_tol"]
        dft.max_cycle = self.opt["max_cyc"]
        dft.grids.level = self.opt["grid"]
        dft.xc = self.opt["method"]
        dft.max_memory = self.opt["memory"]
        if( "guess" in self.opt ):
            dft.init_guess = self.opt["guess"]
        if( bool( self.opt.get( "df", True ) ) ):
            dft = dft.density_fit()
        elif( not self.gpu ):
            dft.direct_scf = True
        # in-memory and on-disk density keep-up (closed and open shell)
        if( self.__dm is None and self.chk ):
            self.__dm = self.__chk_load( aQM )
        # -------------------------------------------------------------------------------
        # embedding of the MM charges into the QM Hamiltonian
        if( len( self.nbn ) > 0 ):
            chg = ( mol.chrg[self.nbn] + self.__dq[self.nbn] ).astype( numpy.float64 )
            if( self.pbc ):
                # Ewald summation requires all the MM charges to be wrapped into
                # the primary cell (here imaged around the QM centroid)
                cen = numpy.mean( mol.coor[self.sel], axis = 0 )
                crd = mol.coor[self.nbn] - mol.boxl * numpy.round( ( mol.coor[self.nbn] - cen ) / mol.boxl, 0 )
                box = numpy.diag( mol.boxl )
                arg = { "unit": "Angstrom" }
                if( self.rad is not None ):
                    arg["radii"] = self.rad
                for itm in [ "rcut_ewald", "rcut_hcore" ]:
                    if( self.opt.get( itm, None ) is not None ):
                        arg[itm] = float( self.opt[itm] )
                if( self.gpu ):
                    scf = gpu4pyscf.qmmm.pbc.itrf.add_mm_charges( dft, crd, box, chg, **arg )
                else:
                    scf = pyscf.qmmm.pbc.itrf.add_mm_charges( dft, crd, box, chg, **arg )
            else:
                crd = self.img_coor( mol )
                if( self.gpu ):
                    scf = gpu4pyscf.qmmm.itrf.add_mm_charges( dft, crd, chg, unit = "Angstrom" )
                elif( self.rad is not None ):
                    scf = pyscf.qmmm.itrf.add_mm_charges( dft, crd, chg, radii = self.rad, unit = "Angstrom" )
                else:
                    scf = pyscf.qmmm.mm_charge( dft, crd, chg, unit = "Angstrom" )
        else:
            scf = dft
        if( not self.gpu ):
            pyscf.lib.num_threads( self.opt["nproc"] )
        return( scf )


    def do_kernel( self, scf ):
        out = float( scf.kernel( dm0 = self.__dm ) ) * self.ce
        if( scf.converged ):
            dm = scf.make_rdm1()
            if( self.gpu ):
                dm = cupy.asnumpy( dm )
            self.__dm = numpy.asarray( dm, dtype = numpy.float64 )
            self.__chk_save()
        else:
            print( ">> pyscf SCF did not converge (%d cycles, conv_tol %g)"%( scf.max_cycle, scf.conv_tol ) )
        return( out )


    def get_func( self, mol ):
        scf = self.update_coor( mol )
        out = self.do_kernel( scf )
        mol.func += out
        chg = scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        return( out )


    def get_grad( self, mol ):
        scf = self.update_coor( mol )
        out = self.do_kernel( scf )
        mol.func += out
        chg = scf.mulliken_pop( verbose = 0 )[1].tolist()
        k = 0
        for i in self.sel:
            mol.chrg[i] = chg[k]
            k += 1
        grd = scf.Gradients()
        grd.verbose = 0
        grd.max_memory = self.opt["memory"]
        if( bool( self.opt.get( "grid_response", True ) ) ):
            grd.grid_response = True
        g = grd.kernel()
        if( self.gpu ):
            g = cupy.asnumpy( g )
        g = numpy.asarray( g, dtype = numpy.float64 ).ravel()
        qm3.engines.link_grad( self.vla, g )
        k = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[k] * self.cg
                k += 1
        for qm_i, mm_j, mm_k, vec in self.vla:
            j3 = mm_j * 3
            for j in [0, 1, 2]:
                mol.grad[mm_k,j] += g[j3+j] * self.cg
        # -------------------------------------------------------------------------------
        # gradient of the embedding term on the MM atoms
        if( len( self.nbn ) > 0 ):
            dm = self.__dm
            g_mm = numpy.asarray( grd.grad_nuc_mm(), dtype = numpy.float64 )
            g_mm += numpy.asarray( grd.grad_hcore_mm( dm ), dtype = numpy.float64 )
            if( self.pbc ):
                g_mm += numpy.asarray( grd.de_ewald_mm, dtype = numpy.float64 )
            for k in range( self.nbn.shape[0] ):
                mol.grad[self.nbn[k],:] += g_mm[k] * self.cg
        return( out )



class runGPU( run ):
    def __init__( self, mol: object,
            opts: typing.Optional[dict] = {},
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        opt = dict( run.__init__.__defaults__[0] )
        opt.update( opts )
        opt["gpu"] = True
        run.__init__( self, mol, opt, sel_QM, sel_MM, link )
