import  numpy
import  typing
import  ctypes
import  os
import  inspect
import  qm3.data


class run( qm3.engines.template ):
    def __init__( self, mol: object, meth: str, chrg: int,
            mult: typing.Optional[int] = 1,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [],
            con: typing.Optional[float] = -1,
            cof: typing.Optional[float] = -1 ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        hami = { "MNDO": 0, "AM1": 1, "RM1": 2, "PM3": 3, "PDDG": 4 }
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 1 + 4 * self.nQM + 4 * self.nMM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_mopac.so" )
        self.lib.qm3_mopac_setup_.argtypes = [ ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_double ), ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_mopac_setup_.restype = None
        self.lib.qm3_mopac_calc_.argtypes = [ ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_mopac_calc_.restype = None
        l = 1
        for i in self.sel:
            self.vec[l] = mol.anum[i]
            l += 1
        for i in range( len( self.lnk ) ):
            self.vec[l] = 1
            l += 1
        self.lib.qm3_mopac_setup_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ),
                ctypes.c_int( hami[meth] ),
                ctypes.c_int( chrg ),
                ctypes.c_int( mult ),
                ctypes.c_int( self.siz ), self.vec,
                ctypes.c_double( con ), ctypes.c_double( cof ) )


    def update_coor( self, mol: object ):
        l = 1
        for i in self.sel:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j]
                l += 1
        self.vla = []
        k = len( self.sel )
        for i in range( len( self.lnk ) ):
            c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            for j in [0, 1, 2]:
                self.vec[l] = c[j]
                l += 1
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            k += 1
        l = 1 + 4 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j] - mol.boxl[j] * numpy.round( mol.coor[i,j] / mol.boxl[j], 0 )
                l += 1
        l  = 1 + 4 * self.nQM
        for i in self.nbn:
            self.vec[l] = mol.chrg[i]
            l += 1


    def get_func( self, mol: object, maxit: typing.Optional[int] = 200 ):
        self.update_coor( mol )
        self.lib.qm3_mopac_calc_( ctypes.c_int( maxit ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1 + 3 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1


    def get_grad( self, mol: object, maxit: typing.Optional[int] = 200 ):
        self.update_coor( mol )
        self.lib.qm3_mopac_calc_( ctypes.c_int( maxit ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1 + 3 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        g = [ self.vec[j] for j in range( 1, 3 * self.nQM + 1 ) ]
        qm3.engines.Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        l = 1 + 4 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1
