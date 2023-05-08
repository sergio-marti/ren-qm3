import  math
import  numpy
import  qm3.maths.grids
import  qm3.utils.pes_samples
import  io


o = qm3.utils.pes_samples.muller_brown()

f = io.StringIO()
lx = numpy.linspace( -1.5, 1.3, 100 )
ly = numpy.linspace( -0.5, 2.0, 100 )
for i in lx:
    for j in ly:
        o.coor = [ i, j ]
        o.get_func()
        f.write( "%20.10lf%20.10lf%20.10lf\n"%( i, j, min( o.func, 50 ) ) )
f.seek( 0 )

g = qm3.maths.grids.grid()
g.parse( f )
g.plot2d()
