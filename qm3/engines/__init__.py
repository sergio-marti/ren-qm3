import  numpy
import  typing


def exclusions( natm, bond, sele ):
    conn = []
    for i in range( natm ):
        conn.append( [] )
    for i,j in bond:
        conn[i].append( j )
        conn[j].append( i )
    excl = []
    nx12 = 0
    nx13 = 0
    nx14 = 0
    for i in numpy.flatnonzero( sele.ravel() ):
        for j in conn[i]:
            if( j != i and not sele[j] ):
                excl.append( [ i, j, 0.0 ] )
                nx12 += 1
            for k in conn[j]:
                if( k != i and not sele[k] ):
                    excl.append( [ i, k, 0.0 ] )
                    nx13 += 1
                for l in conn[k]:
                    if( k != i and l != j and l != i and not sele[l] ):
                        excl.append( [ i, l, 0.5 ] )
                        nx14 += 1
    print( ">> %d exclusions generated (1-2:%d, 1-3:%d, 1-4:%d)"%( nx12 + nx13 + nx14, nx12, nx13, nx14 ) )
    return( excl )


def Link_coor( qm_i, mm_j, mol, dst = 1.1 ):
    vv = mol.coor[mm_j] - mol.coor[qm_i]
    vv /= numpy.linalg.norm( vv )
    vv *= dst
    return( mol.coor[qm_i] + vv, - vv )


def Link_grad( lnk, grd ):
    for qm_i,mm_j,vec in lnk:
        m = numpy.linalg.norm( vec )
        t = sum( [ vec[k] * ( grd[3*qm_i+k] - grd[3*mm_j+k] ) for k in [0, 1, 2] ] ) * 0.5 / m
        grd[3*qm_i:3*qm_i+3] = [ grd[3*qm_i+k] - t * vec[k] / m for k in [0, 1, 2] ]


class template( object ):
    def __init__( self, mol: object,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.flatnonzero( sel_QM )
        else:
            self.sel = numpy.arange( mol.natm )

        # define whether using minimum image for the surrounding charges
        self.img = True
        print( "minimum image NBnd:", self.img )

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
