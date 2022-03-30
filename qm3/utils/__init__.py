import  math
import  numpy
import  typing
import  qm3.data


# =================================================================================================

def from_upper_diagonal( vec: list, by_rows: typing.Optional[bool] = True ) -> numpy.array:
    t = len( vec )
    n = int( round( 0.5 * ( math.sqrt( 1.0 + 8.0 * t ) - 1.0 ), 0 ) )
    if( n * ( n + 1 ) // 2 == t ):
        out = numpy.zeros( ( n, n ) )
        if( by_rows ):
            ii,jj = numpy.triu_indices( n )
        else:
            ii,jj = numpy.tril_indices( n )
        out[ii,jj] = vec
        out[jj,ii] = vec
        return( out )
    else:
        raise Exception( "from_upper_diagonal: invalid dimensions" )

# =================================================================================================

def distanceSQ( ci: numpy.array, cj: numpy.array,
        box: typing.Optional[numpy.array] = numpy.array(
            [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] ) ) -> float:
    vv = ci - cj
    vv -= box * numpy.round( vv / box, 0 )
    return( float( numpy.inner( vv, vv ) ) )


def distance( ci: numpy.array, cj: numpy.array,
        box: typing.Optional[numpy.array] = numpy.array(
            [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] ) ) -> float:
    """
    ci:       array( 3 )
    cj:       array( 3 )
    """
    return( math.sqrt( distanceSQ( ci, cj, box ) ) )


def angleRAD( ci: numpy.array, cj: numpy.array, ck: numpy.array ) -> float:
    vij = ci - cj
    vkj = ck - cj
    try:
        return( math.acos( numpy.dot( vij, vkj ) / ( numpy.linalg.norm( vij ) * numpy.linalg.norm( vkj ) ) ) )
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
        if( math.fabs( o ) > 1.0 ):
            o = math.fabs( o ) / o
        o = math.acos( o ) 
        if( numpy.dot( vij, plk ) < 0.0 ):
            o = -o
    return( float( o ) )


def dihedral( ci: numpy.array, cj: numpy.array, ck: numpy.array, cl: numpy.array ) -> float:
    """
    ci:       array( 3 )
    cj:       array( 3 )
    ck:       array( 3 )
    cl:       array( 3 )
    """
    return( dihedralRAD( ci, cj, ck, cl ) * qm3.data.R2D )

# =================================================================================================

def RT_modes( mol: object ) -> numpy.array:
    size = 3 * mol.actv.sum()
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    mode = numpy.zeros( ( 6, size ), dtype=numpy.float64 )
    cent = numpy.sum( mol.mass * mol.coor * mol.actv, axis = 0 ) / numpy.sum( mol.mass * mol.actv )
    k = 0
    for i in sele:
        sqrm = math.sqrt( mol.mass[i] )
        mode[0,k:k+3] = [ sqrm, 0.0, 0.0 ]
        mode[1,k:k+3] = [ 0.0, sqrm, 0.0 ]
        mode[2,k:k+3] = [ 0.0, 0.0, sqrm ]
        mode[3,k:k+3] = [ 0.0, - ( mol.coor[i,2] - cent[2] ) * sqrm, ( mol.coor[i,1] - cent[1] ) * sqrm ]
        mode[4,k:k+3] = [ ( mol.coor[i,2] - cent[2] ) * sqrm, 0.0, - ( mol.coor[i,0] - cent[0] ) * sqrm ]
        mode[5,k:k+3] = [ - ( mol.coor[i,1] - cent[1] ) * sqrm, ( mol.coor[i,0] - cent[0] ) * sqrm, 0.0 ]
        k += 3
    # orthogonalize modes
    for i in range( 6 ):
        for j in range( i ):
            mode[i] -= numpy.sum( mode[i] * mode[j] ) * mode[j]
        tmp = math.sqrt( numpy.sum( mode[i] * mode[i] ) )
        if( tmp > 0.0 ):
            mode[i] /= tmp
    return( mode )

# =================================================================================================

def k_means( data: numpy.array, K: int ):
    M = [ data[numpy.random.randint(data.shape[0])] ]
    while( len( M ) < K ):
        d2 = numpy.array( [ min( [ numpy.power( numpy.linalg.norm( x - c ), 2.0 ) for c in M ] ) for x in data ] )
        cp = ( d2 / d2.sum() ).cumsum()
        r  = numpy.random.random()
        M.append( data[numpy.where( cp >= r )[0][0]] )
    M = numpy.array( M )
    C = None
    I = None
    o = data[numpy.random.choice( range( data.shape[0] ), K, replace = False )]
    while( C == None or numpy.setdiff1d( numpy.unique( o ), numpy.unique( M ) ).size != 0 ):
        o = M 
        C = {}
        I = {}
        for j in range( data.shape[0] ):
            w = min( [ ( numpy.linalg.norm( data[j] - M[i] ), i ) for i in range( M.shape[0] ) ] )[1]
            try:
                C[w].append( data[j] )
                I[w].append( j )
            except:
                C[w] = [ data[j] ]
                I[w] = [ j ]
        M = numpy.array( [ numpy.mean( C[k], axis = 0 ) for k in iter( C ) ] )
    if( type( C[0][0] ) == numpy.array ):
        C = { k: numpy.array( C[k] ).reshape( ( len( C[k] ), len( C[k][0] ) ) ) for k in iter( C ) }
    else:
        C = { k: numpy.array( C[k] ) for k in iter( C ) }
    return( C, I )



class PCA( object ):
    def __init__( self, data: numpy.array ):
        self.var = data.shape[0]
        self.dim = data.shape[1]
        self.med = data.mean( axis = 1 )
        self.dat = numpy.array( [ data[i,:] - self.med[i] for i in range( self.var ) ] )
        cov = numpy.dot( self.dat, self.dat.T ) / self.dim
        val, vec = numpy.linalg.eigh( cov )
        idx = numpy.argsort( val )
        self.val = val[idx]
        self.vec = vec[:,idx]


    def select( self, sel: list, reduced: typing.Optional[bool] = True ) -> numpy.array:
        ind = sorted( list( set( [ i for i in sel if i >= 0 and i < self.var ] ) ) )
        if( reduced ):
            out = numpy.dot( self.vec[:,ind].T ,self.dat )
            for i in range( len( ind ) ):
                out[i,:] += self.med[ind[i]]
        else:
            out = numpy.dot( numpy.dot( self.vec[:,ind], self.vec[:,ind].T ), self.dat )
            for i in range( self.var ):
                out[i,:] += self.med[i]
        return( out )

# =================================================================================================

def integ_Simpson( x, y ):
    n = len( x )
    o = 0.0
    for i in range( 0, n - 2, 2 ):
        j  = i + 1
        k  = i + 2
        d1 = x[j] - x[i]
        d2 = x[k] - x[j]
        d3 = d1 + d2
        o += ( ( d1 + d1 - d2 ) * y[i] + ( d3 * d3 * y[j] + d1 * ( d2 + d2 - d1 ) * y[k] ) / d2 ) * d3 / 6.0 / d1
    # fix odd intervals with a trapeizodal rule
    if( n%2 != 0 ):
        o += ( y[-1] + y[-2] ) * ( x[-1] - x[-2] ) * 0.5
    return( o )


def integ_GL( f, a, b, n = 80 ):
    x, w = numpy.polynomial.legendre.leggauss( n )
    m = 0.5 * ( b - a )
    p = 0.5 * ( b + a )
    o = 0
    for i in range( n ):
        o += w[i] * f( m * x[i] + p )
    return( o * m )
