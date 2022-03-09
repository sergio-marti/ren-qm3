import  math
import  numpy
import  typing
import  re
import  qm3.data
import  qm3.utils._grids

try:
    import  matplotlib.pyplot
    from mplot3d import axes3d
    from mplot3d import proj3d
    has_mplot3d = True
except:
    has_mplot3d = False

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
    sele = numpy.argwhere( mol.actv.ravel() )
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

class grid( object ):
    __number = re.compile( "^[0-9\.\-eE]+$" )
    def __init__( self ):
        self.x = numpy.array( [] )
        self.y = numpy.array( [] )
        self.z = numpy.array( [] )


    # Whatever the ordering ALWAYS returns: fixed_X, changing_Y
    def parse( self, fdsc: typing.IO, sele: typing.Optional[str] = "0:1:2" ):
        s = [ int( i ) for i in sele.split( ":" ) ]
        d = []
        for l in fdsc:
            t = l.strip().split()
            if( len( t ) >= 3 and self.__number.match( t[s[0]] ) and self.__number.match( t[s[1]] ) and self.__number.match( t[s[2]] ) ):
                d.append( [ float( t[s[0]] ), float( t[s[1]] ), float( t[s[2]] ) ] )
        d.sort()
        ny = 0
        t  = d[ny][0]
        while( d[ny][0] == t ):
            ny += 1
        nz = len( d )
        nx = nz // ny
        self.x = numpy.array( [ d[i*ny][0] for i in range( nx ) ] )
        self.y = numpy.array( [ d[i][1] for i in range( ny ) ] )
        self.z = numpy.array( [ d[i][2] for i in range( nz ) ] )


    # Transform Z into: changing_X, fixed_Y
    def rotate( self ):
        t = []
        k = 0
        for i in self.x:
            for j in self.y:
                t.append( [ j, i, self.z[k] ] )
                k += 1
        t.sort()
        return( numpy.array( [ t[i][2] for i in range( k ) ] ) )


    def regular( self, fdsc: typing.IO,
            points: typing.Optional[tuple] = ( 10, 10 ),
            gauss: typing.Optional[tuple] = ( 0.1, 0.1 ),
            sele: typing.Optional[str] = "0:1:2" ):
        def __pythag( dx, dy ):
            x = math.fabs( dx )
            y = math.fabs( dy )
            if( x > y ):
                return( x * math.sqrt( 1.0 + y * y / ( x * x ) ) )
            if( y == 0.0 ):
                return( 0.0 )
            return( y * math.sqrt( 1.0 + x * x / ( y * y ) ) )
        dat = []
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        s = [ int( i ) for i in sele.split( ":" ) ]
        for l in fdsc:
            t = l.split()
            if( len( t ) >= 3 ):
                if( self.__number.match( t[s[0]] ) and self.__number.match( t[s[1]] ) and self.__number.match( t[s[2]] ) ):
                    rx = float( t[s[0]] )
                    ry = float( t[s[1]] )
                    if( min_x != None and min_y != None and max_x != None and max_y != None ):
                        min_x = min( min_x, rx )
                        min_y = min( min_y, ry )
                        max_x = max( max_x, rx )
                        max_y = max( max_y, ry )
                    else:
                        min_x = rx
                        min_y = ry
                        max_x = rx
                        max_y = ry
                    dat.append( [ rx, ry, float( t[s[2]] ) ] )
        dx = ( max_x - min_x ) / float( points[0] - 1.0 )
        print( "[X] delta: %.4lf  points: %3d  range: %8.2lf / %8.2lf"%( dx, points[0], min_x, max_x ) )
        dy = ( max_y - min_y ) / float( points[1] - 1.0 )
        print( "[Y] delta: %.4lf  points: %3d  range: %8.2lf / %8.2lf"%( dy, points[1], min_y, max_y ) )
        self.x = []
        for i in range( points[0] ):
            self.x.append( min_x + dx * i )
        self.y = []
        for i in range( points[1] ):
            self.y.append( min_y + dy * i )
        try:
            self.z = qm3.utils._grids.regular( self.x, self.y, dat, gauss )
        except:
            self.z = []
            for i in self.x:
                for j in self.y:
                    rz = 0.0
                    rw = 0.0
                    for a,b,c in dat:
                        dst = __pythag( ( a - i ) / gauss[0], ( b - j ) / gauss[1] )
                        w = numpy.exp( - dst * dst )
                        rz += c * w
                        rw += w
                    self.z.append( rz / rw )
        self.x = numpy.array( self.x )
        self.y = numpy.array( self.y )
        self.z = numpy.array( self.z )


    def save( self, fdsc: typing.IO ):
        k = 0
        for i in self.x:
            for j in self.y:
                fdsc.write( "%18.6lf%18.6lf%18.6lf\n"%( i, j, self.z[k] ) )
                k += 1
            fdsc.write( "\n" )


    def plot3d( self, levels: typing.Optional[int] = 20 ):
        """
        For headless terminals set the environment variable:

        export MPLBACKEND=Agg
        """
        if( has_mplot3d ):
            def __orthogonal_proj(zfront, zback):
                a = (zfront+zback)/(zfront-zback)
                b = -2*(zfront*zback)/(zfront-zback)
                return numpy.array([[1,0,0,0], [0,1,0,0], [0,0,a,b], [0,0,-0.0001,zback]])
            proj3d.persp_transformation = __orthogonal_proj
            fig = matplotlib.pyplot.figure()
            axs = fig.gca( projection = "3d" )
#            axs = axes3d.Axes3D( matplotlib.pyplot.figure() )
            rz  = self.rotate()
            nx  = len( self.x )
            ny  = len( self.y )
            lx  = []
            ly  = []
            lz  = []
            for i in range( ny ):
                lx.append( self.x[:] )
                ly.append( nx * [ self.y[i] ] )
                lz.append( rz[i*nx:(i+1)*nx][:] )
            z_min = min( self.z )
            z_max = max( self.z )
            z_lvl = [ z_min + ( z_max - z_min ) / float( levels ) * float( i ) for i in range( levels + 1 ) ]
            lz = numpy.array( lz, dtype=numpy.float64 )
            axs.plot_surface( lx, ly, lz, rstride = 1, cstride = 1, cmap = "coolwarm", linewidths = 0.1 )
#            axs.contour( lx, ly, lz, zdir = "z", levels = z_lvl, linewidths = 2, cmap = "coolwarm" )
            axs.contour( lx, ly, lz, zdir = "z", offset = z_min, levels = z_lvl, linewidths = 2, cmap = "coolwarm" )
            axs.view_init( 90, -89 )
            matplotlib.pyplot.show()
        else:
            return


    def plot2d( self, levels: typing.Optional[int] = 20, fname: typing.Optional[str] = None ):
        """
        For headless terminals set the environment variable:

        export MPLBACKEND=Agg
        """
        if( has_mplot3d ):
            rz  = self.rotate()
            nx  = len( self.x )
            ny  = len( self.y )
            lx  = []
            ly  = []
            lz  = []
            for i in range( ny ):
                lx.append( self.x[:] )
                ly.append( nx * [ self.y[i] ] )
                lz.append( rz[i*nx:(i+1)*nx][:] )
            z_min = min( self.z )
            z_max = max( self.z )
            z_lvl = [ z_min + ( z_max - z_min ) / float( levels ) * float( i ) for i in range( levels + 1 ) ]
            matplotlib.pyplot.contourf( lx, ly, lz, levels = z_lvl, cmap = "coolwarm" )
            cntr = matplotlib.pyplot.contour( lx, ly, lz, levels = z_lvl, colors = "black", linewidths = 2 )
            matplotlib.pyplot.clabel( cntr, inline = True, levels = z_lvl, fontsize = 7, fmt = "%.1lf" )
            if( fname != None ):
                matplotlib.pyplot.savefig( fname )
            matplotlib.pyplot.show()
        else:
            return

