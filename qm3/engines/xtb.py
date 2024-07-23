import  numpy
import  typing
import  ctypes
import  os
import  inspect
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, chrg: int,
            nope: typing.Optional[int] = 0,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 3 + 5 * self.nQM + 4 * self.nMM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_xtb.so" )
        self.lib.qm3_xtb_calc_.argtypes = [ 
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_xtb_calc_.restype = None
        self.vec[1] = chrg
        self.vec[2] = nope
        l = 3
        for i in self.sel:
            self.vec[l] = mol.anum[i]
            l += 1
        for i in range( len( self.lnk ) ):
            self.vec[l] = 1
            l += 1
#        l  = 3 + 5 * self.nQM
#        for i in self.nbn:
#            self.vec[l] = mol.chrg[i]
#            l += 1
        # redistribute MM-charge on the remaining atoms of the group
        self.__dq = numpy.zeros( mol.natm )
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------


    def update_coor( self, mol ):
        l = 3 + self.nQM
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
        l = 3 + 5 * self.nQM + self.nMM
        for i in self.nbn:
            tmp = mol.coor[i]
            if( self.img ):
                tmp -= mol.boxl * numpy.round( tmp / mol.boxl, 0 )
            for j in [0, 1, 2]:
                self.vec[l] = tmp[j]
                l += 1
        l  = 3 + 5 * self.nQM
        for i in self.nbn:
            self.vec[l] = mol.chrg[i] + self.__dq[i]
            l += 1


    def get_func( self, mol, density = False ):
        self.update_coor( mol )
        self.lib.qm3_xtb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 3 + 4 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        return( self.vec[0] )


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_xtb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 3 + 4 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        l = 3 + self.nQM
        g = [ self.vec[l+j] for j in range( 3 * self.nQM ) ]
        qm3.engines.Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        l = 3 + 5 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1
        return( self.vec[0] )
