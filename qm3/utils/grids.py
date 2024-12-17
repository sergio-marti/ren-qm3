import  math
import  numpy
import  typing
import  re
import  qm3.data
import  qm3.utils._grids

try:
    import  matplotlib.pyplot
    has_pyplot = True
except:
    has_pyplot = False


class grid( object ):
    __number = re.compile( "^[0-9\\.\\-eE+]+$" )
    def __init__( self ):
        self.x = numpy.array( [] )
        self.y = numpy.array( [] )
        self.z = numpy.array( [] )


    # Whatever the ordering ALWAYS returns: fixed_X, changing_Y
    def parse( self, fdsc: typing.IO, sele: typing.Optional[str] = "0:1:2" ):
        s = [ int( i ) for i in sele.split( ":" ) ]
        d = []
        for l in fdsc:
            t = l.split()
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
        if( has_pyplot ):
            ly, lx = numpy.meshgrid( self.y, self.x )
            lz = self.z.reshape( ( self.x.shape[0], self.y.shape[0] ) )
            z_min = min( self.z )
            z_max = max( self.z )
            z_lvl = [ z_min + ( z_max - z_min ) / float( levels ) * float( i ) for i in range( levels + 1 ) ]
            axs = matplotlib.pyplot.axes( projection = "3d" )
            axs.plot_surface( lx, ly, lz, rstride = 1, cstride = 1, cmap = "coolwarm", linewidths = 0.1 )
            axs.contour( lx, ly, lz, zdir = "z", offset = z_min, levels = z_lvl, linewidths = 2, cmap = "coolwarm" )
            axs.view_init( 90, -90 )
            matplotlib.pyplot.show()
        else:
            return


    def plot2d( self, levels: typing.Optional[int] = 20, fname: typing.Optional[str] = None ):
        """
        For headless terminals set the environment variable:

        export MPLBACKEND=Agg
        """
        if( has_pyplot ):
            ly, lx = numpy.meshgrid( self.y, self.x )
            lz = self.z.reshape( ( self.x.shape[0], self.y.shape[0] ) )
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

