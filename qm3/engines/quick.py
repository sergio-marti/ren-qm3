import  numpy
import  typing
import  ctypes
import  os
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, keys: str,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [],
            library: typing.Optional[str] = "libquick.so",
            info_mpi: typing.Optional[tuple] = () ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        # ------------------------------------------------------------------------
        self.__err = ( ctypes.c_int )()
        self.lib = ctypes.CDLL( library )
        if( info_mpi != () ):
            self.lib_fixmpi = self.lib["__quick_api_module_MOD_set_quick_mpi"]
            self.lib_fixmpi.argtypes = [ 
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ) ]
            self.lib_fixmpi.restype = None
            self.lib_fixmpi( ctypes.c_int( info_mpi[0] ), ctypes.c_int( info_mpi[1] ), self.__err )
        self.lib_setup = self.lib["__quick_api_module_MOD_set_quick_job"]
        self.lib_setup.argtypes = [ 
            ctypes.POINTER( ctypes.c_char ),
            ctypes.POINTER( ctypes.c_char ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_bool ),
            ctypes.POINTER( ctypes.c_int ) ]
        self.lib_setup.restype = None
        self.lib_calc = self.lib["__quick_api_module_MOD_get_quick_energy_gradients"]
        self.lib_calc.argtypes = [ 
            ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_int ) ]
        self.lib_calc.restype = None
        # ------------------------------------------------------------------------
        self.__out = ( ctypes.c_char * 80 )()
        self.__out.value = b"quick.out "
        self.__cmd = ( ctypes.c_char * 256 )()
        if( self.nMM > 0 ):
            self.__cmd.value = keys.encode( "utf-8" ) + b" gradient dipole extcharges "
        else:
            self.__cmd.value = keys.encode( "utf-8" ) + b" gradient dipole "
        anum = ( ctypes.c_int * self.nQM )()
        l = 0
        for i in self.sel:
            anum[l] = mol.anum[i]
            l += 1
        for i in range( len( self.lnk ) ):
            anum[l] = 1
            l += 1
        self.lib_setup( self.__out, self.__cmd,
                ctypes.c_int( self.nQM ), anum, ctypes.c_bool( True ), self.__err )
        # redistribute MM-charge on the remaining atoms of the group
        self.__dq = numpy.zeros( mol.natm )
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------


    def get_grad( self, mol: object ):
        func = ( ctypes.c_double )()
        func.value = 0.0
        qcrd = numpy.zeros( ( self.nQM, 3 ) )
        qgrd = numpy.zeros( ( self.nQM, 3 ) ).ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
        mcrd = numpy.zeros( ( self.nMM, 4 ) )
        mgrd = numpy.zeros( ( self.nMM, 3 ) ).ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
        l = 0
        for i in self.sel:
            qcrd[l,:] = mol.coor[i,:]
            l += 1
        k = len( self.sel )
        self.vla = []
        for i in range( len( self.lnk ) ):
            c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            qcrd[l,:] = c[:]
            l += 1
            k += 1
        l = 0
        for i in self.nbn:
            tmp = mol.coor[i]
            if( self.img ):
                tmp -= mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
            mcrd[l,0:3] = tmp
            mcrd[l,3] = mol.chrg[i] + self.__dq[i]
            l += 1
        self.lib_calc( 
            qcrd.ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
            ctypes.c_int( self.nMM ), 
            mcrd.ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
            func, qgrd, mgrd, self.__err )
        out = func.value * self.ce
        mol.func += out
        qgrd = numpy.ctypeslib.as_array( qgrd, ( 3, self.nQM ) ).ravel() * self.cg
        qm3.engines.Link_grad( self.vla, qgrd )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += qgrd[l]
                l += 1
        mgrd = numpy.ctypeslib.as_array( mgrd, ( 3, self.nMM ) ).ravel() * self.cg
        l = 0
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += mgrd[l]
                l += 1
        return( out )

