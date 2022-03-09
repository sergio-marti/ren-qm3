import  numpy
import  typing


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
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) )
            for i,j in link:
                self.nbn[j] = False
            self.nbn = numpy.argwhere( self.nbn ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
