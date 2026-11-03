import  numpy
import  typing


def exclusions( natm, bond, sele ):
    conn = [ [] for i in range( natm ) ]
    for i,j in bond:
        conn[i].append( j )
        conn[j].append( i )
    excl = []
    nx12 = 0
    nx13 = 0
    nx14 = 0
    for i in numpy.flatnonzero( sele.ravel() ):
        seen12 = set()
        seen13 = set()
        seen14 = set()
        for j in conn[i]:
            if( j != i and j not in seen12 and not sele[j] ):
                seen12.add( j )
                excl.append( [ int( i ), j, 0.0 ] )
                nx12 += 1
            for k in conn[j]:
                if( k != i and k not in seen12 and k not in seen13 and not sele[k] ):
                    seen13.add( k )
                    excl.append( [ int( i ), k, 0.0 ] )
                    nx13 += 1
                for l in conn[k]:
                    if( k != i and l != j and l != i and l not in seen12 and l not in seen13 and l not in seen14 and not sele[l] ):
                        seen14.add( l )
                        excl.append( [ int( i ), l, 0.5 ] )
                        nx14 += 1
    print( ">> %d exclusions generated (1-2:%d, 1-3:%d, 1-4:%d)"%( nx12 + nx13 + nx14, nx12, nx13, nx14 ) )
    return( excl )


def link_coor( qm_i, mm_j, mol, dst = 1.1 ):
    r = mol.coor[mm_j] - mol.coor[qm_i]
    return( mol.coor[qm_i] + r / numpy.linalg.norm( r ) * dst, r )


def link_grad( lnk, grd, dst = 1.1 ):
    for qm_i, mm_j, mm_k, vec in lnk:
        r = numpy.linalg.norm( vec )
        u = vec / r
        J = ( dst / r ) * ( numpy.eye( 3 ) - numpy.outer( u, u ) )
        g_la = numpy.array( grd[3*mm_j:3*mm_j+3].copy() )
        g_M  = J.dot( g_la )
        g_Q  = g_la - g_M
        grd[3*qm_i:3*qm_i+3] = [ grd[3*qm_i+k] + g_Q[k] for k in [0, 1, 2] ]
        grd[3*mm_j:3*mm_j+3] = g_M.tolist()


class template( object ):
    def __init__( self, mol: object,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.flatnonzero( sel_QM )
        else:
            self.sel = numpy.arange( mol.natm )

        # define whether using minimum image convention for the surrounding charges
        self.img = False
        self.img_cell = None

        self.vla = []
        self.lnk = []
        self.grp = {}
        for i in range( len( link ) ):
            if( len( link[i] ) == 3 ):
                self.lnk.append( ( link[i][0], link[i][1] ) )
                self.grp[link[i][1]] = link[i][2][:]
            else:
                self.lnk.append( link[i] )
        if( len( self.lnk ) > 0 ):
            print( "link-atoms [QM,MM]:", self.lnk )
            print( "link-charge groups:", self.grp )

        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) )
            for i,j in self.lnk:
                self.nbn[j] = False
            self.nbn = numpy.flatnonzero( self.nbn )
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )


    def __setattr__( self, name, value ):
        if( name == "img" ):
            print( "minimum image NBnd:", value )
        super( template, self ).__setattr__( name, value )


    def img_init( self, mol: object,
            cen: typing.Optional[numpy.array] = None,
            img_file: typing.Optional[str] = "img_cell.npy" ):
        self.img = True
        try:
            self.img_cell = numpy.load( img_file )
            print( "minumum image info loaded from:", img_file )
        except:
            crd = mol.coor[self.nbn].copy()
            if( cen is None or cen.shape[0] != 3 ):
                cen = numpy.mean( mol.coor[self.sel], axis = 0 )
            self.img_cell = numpy.round( ( crd - cen ) / mol.boxl, 0 )
            numpy.save( img_file, self.img_cell )
            print( "minumum image info saved to:", img_file )


    def img_coor( self, mol: object ):
        crd = mol.coor[self.nbn].copy()
        if( self.img ):
            crd -= mol.boxl * self.img_cell
        return( crd )


    def img_clean( self ):
        self.img = False
        self.img_cell = None
