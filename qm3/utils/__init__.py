import  numpy
import  typing
import  qm3.data


def distanceSQ( ci: numpy.array, cj: numpy.array,
        box: typing.Optional[numpy.array] = numpy.array(
            [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] ) ) -> float:
    vv = ci - cj
    vv -= box * numpy.round( vv / box, 0 )
    return( numpy.dot( vv, vv ) )


def distance( ci: numpy.array, cj: numpy.array,
        box: typing.Optional[numpy.array] = numpy.array(
            [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] ) ) -> float:
    """
    ci:       array( 3 )
    cj:       array( 3 )
    """
    return( numpy.sqrt( distanceSQ( ci, cj, box ) ) )


def angleRAD( ci: numpy.array, cj: numpy.array, ck: numpy.array ) -> float:
    vij = ci - cj
    vkj = ck - cj
    try:
        return( numpy.arccos( numpy.dot( vij, vkj ) / ( numpy.linalg.norm( vij ) * numpy.linalg.norm( vkj ) ) ) )
    except:
        raise Exception( "utils.angleRAD: invalid angle" )


def angle( ci: numpy.array, cj: numpy.array, ck: numpy.array ) -> float:
    """
    ci:       array( 3 )
    cj:       array( 3 )
    ck:       array( 3 )
    """
    return( angleRAD( ci, cj, ck ) * qm3.data.R2D )


def dihedralRAD( ci: numpy.array, cj: numpy.array, ck: numpy.array, cl: numpy.array ) -> float:
    vij = ci - cj
    vkj = ck - cj
    vlj = cl - cj
    pik = numpy.cross( vij, vkj )
    plk = numpy.cross( vlj, vkj )
    m1  = numpy.dot( pik, plk )
    m2  = numpy.linalg.norm( pik )
    m3  = numpy.linalg.norm( plk )
    o   = 0.0
    if( m2 != 0.0 and m3 != 0.0 ):
        o = m1 / ( m2 * m3 )
        if( numpy.fabs( o ) > 1.0 ):
            o = numpy.fabs( o ) / o
        o = numpy.arccos( o ) 
        if( numpy.dot( vij, plk ) < 0.0 ):
            o = -o
    return( o )


def dihedral( ci: numpy.array, cj: numpy.array, ck: numpy.array, cl: numpy.array ) -> float:
    """
    ci:       array( 3 )
    cj:       array( 3 )
    ck:       array( 3 )
    cl:       array( 3 )
    """
    return( dihedralRAD( ci, cj, ck, cl ) * qm3.data.R2D )
