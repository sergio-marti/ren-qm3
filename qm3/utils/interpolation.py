import  numpy
import  typing



def find_center( rx: float, x: numpy.array ) -> int:
    try:
        w = numpy.where( x <= rx )[0][-1]
    except:
        w = 0
    if( w == len( x ) ):
        return( w - 1 )
    else:
        return( w )
        


class gaussian( object ):
    def __init__( self, x: numpy.array, y: numpy.array,
            sigma: typing.Optional[float] = 0.1 ):
        self.x = x.copy()
        self.y = y.copy()
        self.sigma = sigma


    def calc( self, rx: float ) -> tuple:
        w  = numpy.exp( - numpy.power( ( rx - self.x ) / self.sigma, 2 ) )
        t  = numpy.sum( w )
        ry = numpy.sum( self.y * w ) / t
        dy = numpy.sum( self.y * w * 2.0 * ( rx - self.x ) / self.sigma ) / t
        return( ry, - dy )



class cubic_spline( object ):
    def __init__( self, x: numpy.array, y: numpy.array ):
        t = numpy.argsort( x )
        self.x = x[t]
        self.y = y[t]
        self.n = len( x )
        self.y2 = numpy.zeros( self.n )
        u = numpy.zeros( self.n )
        u[0]  = 0.0
        u[-1] = 0.0
        self.y2[0]  = 0.0
        self.y2[-1] = 0.0
        for i in range( 1, self.n - 1 ):
            s = ( self.x[i] - self.x[i-1] ) / ( self.x[i+1] - self.x[i-1] )
            p = s * self.y2[i-1] + 2.0
            self.y2[i] = ( s - 1.0 ) / p
            u[i]=( 6.0 * ( ( self.y[i+1] - self.y[i] ) / ( self.x[i+1] - self.x[i] ) - ( self.y[i] - self.y[i-1] ) / ( self.x[i] - self.x[i-1] ) ) / ( self.x[i+1] - self.x[i-1] ) - s * u[i-1] ) / p
        for i in range( self.n-2, -1, -1 ):
            self.y2[i] = self.y2[i] * self.y2[i+1] + u[i]

    
    def calc( self, rx: float ) -> tuple:
        klo = max( 0, find_center( rx, self.x ) )
        khi = min( self.n - 1, klo + 1 )
        h   = self.x[khi] - self.x[klo]
        a   = ( self.x[khi] - rx ) / h
        b   = ( rx - self.x[klo] ) / h
        ry  = a * self.y[klo] + b * self.y[khi] + ( ( a * a * a - a ) * self.y2[klo] + ( b * b * b - b ) * self.y2[khi] ) * ( h * h ) / 6.0
        dy  = ( self.y[khi] - self.y[klo] ) / h + h * ( ( 3.0 * b * b - 1.0 ) * self.y2[khi] - ( 3.0 * a * a - 1.0 ) * self.y2[klo] ) / 6.0
        return( ry, dy )



class hermite_spline( object ):
    """
    Available methods: steffen  /  akima  /  [fritsch_carlson]
    """
    def __init__( self, x: numpy.array, y: numpy.array,
            method: typing.Optional[str] = "fritsch_carlson" ):
        t = numpy.argsort( x )
        self.x = x[t]
        self.y = y[t]
        self.n = len( x )
        dx = numpy.ediff1d( self.x )
        dy = numpy.ediff1d( self.y )
        m  = dy / dx
        self.c1 = []
        self.c2 = []
        self.c3 = []
        # -------------------------------------------------------------------
        # Steffen
        if( method == "steffen" ):
            self.c1.append( m[0] )
            for i in range( self.n - 2 ):
                self.c1.append( ( numpy.copysign( 1.0, m[i] ) + numpy.copysign( 1.0, m[i+1] ) ) * min( numpy.fabs( m[i] ), numpy.fabs( m[i+1] ), 0.5 * numpy.fabs( ( dx[i] * m[i+1] + dx[i+1] * m[i] ) / ( dx[i] + dx[i+1] ) ) ) )
            self.c1.append( m[-1] )
        # -------------------------------------------------------------------
        # Akima
        elif( method == "akima" ):
            M  = [ 2.0 * m[0] - m[1], 2.0 * m[0] - m[1] ] + m.tolist()
            M += [ 2.0 * m[-1] - m[-2], 2.0 * ( 2.0 * m[-1] - m[-2] ) - m[-1] ]
            for i in range( self.n ):
                a = numpy.fabs( M[i+3] - M[i+2] )
                b = numpy.fabs( M[i+1] - M[i] )
                if( a+b > 0.0 ):
                    self.c1.append( ( b * M[i+2] + a * M[i+1] ) / ( a + b ) )
                else:
                    self.c1.append( ( M[i+2] + M[i+1] ) / 2.0 )
        # -------------------------------------------------------------------
        # Fritsch-Carlson
        else:
            self.c1.append( m[0] )
            for i in range( self.n - 2 ):
                if( m[i] * m[i+1] <= 0.0 ):
                    self.c1.append( 0.0 )
                else:
                    t = dx[i] + dx[i+1]
                    self.c1.append( 3.0 * t / ( ( t + dx[i+1] ) / m[i] + ( t + dx[i] ) / m[i+1] ) )
            self.c1.append( m[-1] )
        # -------------------------------------------------------------------
        for i in range( self.n - 1 ):
            t = self.c1[i] + self.c1[i+1] - m[i] - m[i]
            self.c2.append( ( m[i] - self.c1[i] - t ) / dx[i] )
            self.c3.append( t / ( dx[i] * dx[i] ) )


    def calc( self, rx: float ) -> tuple:
        i  = find_center( rx, self.x )
        h  = rx - self.x[i]
        h2 = h * h
        ry = self.y[i] + ( self.c1[i] + self.c2[i] * h + self.c3[i] * h2 ) * h
        dy = self.c1[i] + ( 2.0 * self.c2[i] + 3.0 * self.c3[i] * h ) * h
        return( ry, dy )



class interpolate_2d( object ):
    """
    Z data should be porperly sorted as fixed_X, changing_Y
    
    Interpolant can make use of a lambda function:

            interpolant = lambda x,y: qm3.utils.interpolation.hermite_spline( x, y, method = "akima" )
    """
    def __init__( self, x: numpy.array, y: numpy.array, z: numpy.array,
            interpolant: typing.Optional[typing.Callable] = cubic_spline ):
        self.nx = len( x )
        self.ny = len( y )
        self.x  = x
        self.y  = y
        self.z  = z
        self.II = interpolant
        self.Ix = []
        for i in range( self.nx ):
            self.Ix.append( self.II( self.y, numpy.array( [ self.z[self.ny*i+j] for j in range( self.ny ) ] ) ) )
        self.Iy = []
        for j in range( self.ny ):
            self.Iy.append( self.II( self.x, numpy.array( [ self.z[self.ny*i+j] for i in range( self.nx ) ] ) ) )


    def calc( self, rx: float, ry: float ) -> tuple:
        ox = self.II( self.x, numpy.array( [ self.Ix[i].calc( ry )[0] for i in range( self.nx ) ] ) ).calc( rx )
        oy = self.II( self.y, numpy.array( [ self.Iy[i].calc( rx )[0] for i in range( self.ny ) ] ) ).calc( ry )
        return( ( ox[0] + oy[0] ) * 0.5, ox[1], oy[1] )
