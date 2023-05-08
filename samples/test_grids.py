import  math
import  numpy
import  qm3.utils.grids
import  io


def muller_brown( x, y ):
    A  = [ -200.0, -100.0, -170.0, 15.0 ]
    a  = [ -1.0, -1.0, -6.5, 0.7 ]
    b  = [ 0.0, 0.0, 11.0, 0.6 ]
    c  = [ -10.0, -10.0, -6.5, 0.7 ]
    xo = [ 1.0, 0.0, -0.5, -1.0 ]
    yo = [ 0.0, 0.5, 1.5, 1.0 ]
    rr = 0.0
    for i in range( 4 ):
        rr += A[i] * math.exp( a[i] * math.pow( x - xo[i], 2.0 ) + b[i] * ( x - xo[i] ) * ( y - yo[i] ) + c[i] * math.pow( y - yo[i], 2.0 ) )
    return( rr )


f = io.StringIO()
lx = numpy.linspace( -1.5, 1.3, 100 )
ly = numpy.linspace( -0.5, 2.0, 100 )
for i in lx:
    for j in ly:
        f.write( "%20.10lf%20.10lf%20.10lf\n"%( i, j, min( muller_brown( i , j ), 50 ) ) )
f.seek( 0 )

g = qm3.utils.grids.grid()
g.parse( f )
g.plot2d()
