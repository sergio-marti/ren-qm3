import  numpy
import  typing
import  ctypes
import  os
import  inspect
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, fdsc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 1 + 3 * ( self.nQM + self.nMM ) + self.nQM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_sqm.so" )
        self.lib.qm3_sqm_calc_.argtypes = [ ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_sqm_calc_.restype = None
        self.inp = fdsc.read()
        self.mk_input( mol )
        self.lib.qm3_sqm_init_()


    def mk_input( self, mol ):
        s_qm = ""
        j = 0
        for i in self.sel:
            s_qm += "%3d%4s%20.10lf%20.10lf%20.10lf\n"%( mol.anum[i], qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
            j += 1
        dq = numpy.zeros( mol.natm )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                s_qm += "%3d%4s%20.10lf%20.10lf%20.10lf\n"%( 1, "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            # redistribute MM-charge on the remaining atoms of the group
            for i,j in self.lnk:
                if( j in self.grp ):
                    dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
            # ----------------------------------------------------------
        s_mm = ""
        if( len( self.nbn ) > 0 ):
            s_mm = "#EXCHARGES\n"
            for i in self.nbn:
                tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
                s_mm += "%3d%4s%20.10lf%20.10lf%20.10lf%12.4lf\n"%( 1, "H",
                        tmp[0], tmp[1], tmp[2], mol.chrg[i] + dq[i] )
            s_mm += "#END"
        f = open( "sqm_mdin", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm )
        buf = buf.replace( "qm3_charges", s_mm )
        f.write( buf )
        f.close()


    def update_coor( self, mol ):
        l = 0
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
        l = 3 * self.nQM
        for i in self.nbn:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j] - mol.boxl[j] * numpy.round( mol.coor[i,j] / mol.boxl[j], 0 )
                l += 1


    def get_func( self, mol, density = False ):
        self.update_coor( mol )
        self.lib.qm3_sqm_calc_( ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_sqm_calc_( ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        l = 1 + self.nQM
        g = [ self.vec[l+j] for j in range( 3 * self.nQM ) ]
        qm3.engines.Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        l = 1 + 4 * self.nQM
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1
