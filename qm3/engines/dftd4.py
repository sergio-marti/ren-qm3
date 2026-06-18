import  numpy
import  typing
import  ctypes
import  os
import  inspect
import  qm3.data
import  qm3.engines


class run( object ):
    def __init__( self, mol: object, parm: dict,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.flatnonzero( sel_QM )
        else:
            self.sel = numpy.arange( mol.natm )
        self.lnk = link[:]
        self.vla = []
        self.nat = len( self.sel )
        self.all = self.nat + len( self.lnk )
        self.siz = max( 6, self.all * 3 + 1 )
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_dftd4.so" )
        self.lib.qm3_dftd4_init_.argtypes = [
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_dftd4_init_.restype = None
        self.lib.qm3_dftd4_calc_.argtypes = [
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_dftd4_calc_.restype = None
        self.vec[0] = parm["chrg"]
        k = 1
        for i in self.sel:
            self.vec[k] = mol.anum[i]
            k += 1
        if( len( self.lnk ) > 0 ):
            for i in range( len( self.lnk ) ):
                self.vec[k] = 1
                k += 1
        self.vec[self.all+1] = parm["s6"]
        self.vec[self.all+2] = parm["s8"]
        self.vec[self.all+3] = parm["a1"]
        self.vec[self.all+4] = parm["a2"]
        self.lib.qm3_dftd4_init_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )


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


    def get_func( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_dftd4_calc_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        return( self.vec[0] )


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_dftd4_calc_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        g = [ j for j in self.vec[1:] ]
        qm3.engines.Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        return( self.vec[0] )




try:
    import  dftd4.interface
    import  atexit

    class native( object ):
        def __init__( self, mol: object, param, chrg: typing.Optional[int] = 0,
                sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
                link: typing.Optional[list] = [] ):
            self.cx  = 1.0 / qm3.data.A0
            self.ce  = qm3.data.H2J
            self.cg  = self.ce / qm3.data.A0
            if( type( param ) == str ):
                self.prm = dftd4.interface.DampingParam( method = param, atm = True )
            else:
                self.prm = param
            if( sel_QM.sum() > 0 ):
                self.sel = numpy.flatnonzero( sel_QM )
            else:
                self.sel = numpy.arange( mol.natm )
            self.lnk = link[:]
            anu = []
            crd = []
            for i in self.sel:
                crd.append( mol.coor[i] * self.cx )
                anu.append( mol.anum[i] )
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                crd.append( c * self.cx )
                anu.append( 1 )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            self.mdl = dftd4.interface.DispersionModel( numbers = numpy.array( anu ), positions = numpy.array( crd ), charge = chrg )
            atexit.register( self.clean )


        def clean( self ):
            del( self.prm )
            del( self.mdl )


        def update_coor( self, mol ):
            crd = []
            for i in self.sel:
                crd.append( mol.coor[i] * self.cx )
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                crd.append( c * self.cx )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            self.mdl.update( positions = numpy.array( crd ) )

    
        def get_func( self, mol ):
            self.update_coor( mol )
            out = self.mdl.get_dispersion( self.prm, grad = False )
            ene = float( out["energy"] ) * self.ce
            mol.func += ene
            return( ene )
    
    
        def get_grad( self, mol ):
            self.update_coor( mol )
            out = self.mdl.get_dispersion( self.prm, grad = True )
            ene = float( out["energy"] ) * self.ce
            mol.func += ene
            g = out["gradient"].ravel() * self.cg
            qm3.engines.Link_grad( self.vla, g )
            l = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[l]
                    l += 1
            return( ene )


except:
    pass
