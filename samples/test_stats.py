import  numpy
import  qm3.utils.stats
import  matplotlib.pyplot as plt

x = numpy.array( [ 7, 4, 10, 16, 13, 7, 3, 5, 7, 3, 13, 14, 12, 11, 10, 7, 7, 5, 3, 3 ] )
print( qm3.utils.stats.k_means( x, 2 )[0] )
print( 80 * "=" )

x.shape = ( 4, 5 )
o = qm3.utils.stats.PCA( x )
print( o.val )
print( 80 * "-" )
s = numpy.array( [ 1, 2, 3 ] )
print( o.select( s ) )
print( 80 * "-" )
print( o.select( s, False ) )

numpy.random.seed()
x = numpy.linspace( 0, 2 * numpy.pi, 100 )
y = numpy.sin( x )
p = y + numpy.random.random( 100 ) - 0.5
f, g = qm3.utils.stats.savitzky_golay( x, p )

plt.plot( x, y, '.' )
plt.plot( x, p, 'o' )
plt.plot( x, f, '-' )
plt.show()
