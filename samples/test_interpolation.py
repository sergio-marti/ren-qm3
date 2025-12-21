import  numpy
import  matplotlib.pyplot as plt
import  qm3.utils.interpolation

x = numpy.linspace( 0, 10, 101 )
y = numpy.sin( x )

gg = qm3.utils.interpolation.gaussian( x, y, 0.005 )
cs = qm3.utils.interpolation.cubic_spline( x, y )
ss = qm3.utils.interpolation.hermite_spline( x, y, "steffen" )
aa = qm3.utils.interpolation.hermite_spline( x, y, "akima" )
fc = qm3.utils.interpolation.hermite_spline( x, y, "fritsch_carlson" )

r = 3.05
print( "Real:     ", r, numpy.sin( r ), numpy.cos( r ) )
print( "Gaussian: ", gg.calc( r ) )
print( "CSpline:  ", cs.calc( r ) )
print( "Steffen:  ", ss.calc( r ) )
print( "Akima:    ", aa.calc( r ) )
print( "Fritsch.C:", fc.calc( r ) )
print( 80*'-' )

r = ( x[0] + x[1] ) * 0.5
print( "Real:     ", r, numpy.sin( r ), numpy.cos( r ) )
print( "Gaussian: ", gg.calc( r ) )
print( "CSpline:  ", cs.calc( r ) )
print( "Steffen:  ", ss.calc( r ) )
print( "Akima:    ", aa.calc( r ) )
print( "Fritsch.C:", fc.calc( r ) )
print( 80*'-' )

r = ( x[-1] + x[-2] ) * 0.5
print( "Real:     ", r, numpy.sin( r ), numpy.cos( r ) )
print( "Gaussian: ", gg.calc( r ) )
print( "CSpline:  ", cs.calc( r ) )
print( "Steffen:  ", ss.calc( r ) )
print( "Akima:    ", aa.calc( r ) )
print( "Fritsch.C:", fc.calc( r ) )

print( 80 * '-' )
x = numpy.array( [ 1., 2., 3., 4., 5., 6., 7., 8., 9., 10. ] )
y = numpy.array( [ 10., 10., 10., 10., 10., 10., 10.5, 15., 50., 60., 85. ] )
print( "CSpline:  ", qm3.utils.interpolation.cubic_spline( x, y ).calc( 9.5 ) )
print( "Fritsch.C:", qm3.utils.interpolation.hermite_spline( x, y ).calc( 9.5 ) )
print( "Steffen:  ", qm3.utils.interpolation.hermite_spline( x, y, "steffen" ).calc( 9.5 ) )
print( "Akima:    ", qm3.utils.interpolation.hermite_spline( x, y, "akima" ).calc( 9.5 ) )


numpy.random.seed()
x = numpy.linspace( 0, 2 * numpy.pi, 100 )
y = numpy.sin( x )
p = y + numpy.random.random( 100 ) - 0.5
f = qm3.utils.interpolation.savitzky_golay( p )
g = qm3.utils.interpolation.modified_sinc_filter( p, p.shape[0], 2, 4.0 )
plt.plot( x, y, '.' )
plt.plot( x, p, 'o' )
plt.plot( x, f, '-', label = "savitzky_golay" )
plt.plot( x, g, '-', label = "modified_sinc_filter(n,2,4)" )
plt.legend()
plt.show()
