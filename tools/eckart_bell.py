#!/usr/bin/env python3
import  sys
import  math
import  numpy
import  matplotlib.pyplot as plt

#10.1098/rspa.1933.0031

if( len( sys.argv ) != 6 ):
    print( "%s  Er_kJ/mol  Et_kJ/mol  Ep_kJ/mol  ifreq_cm^-1  Temp_K"%( sys.argv[0] ) )
    sys.exit(1)

cte = 120.2724
pi2 = 2.0 * math.pi
e_r = float( sys.argv[1] )
e_t = float( sys.argv[2] )
e_p = float( sys.argv[3] )
frq = float( sys.argv[4] )
tem = float( sys.argv[5] )
act = e_t - e_r
print( "E_r  ", round( e_r, 2 ), "_kJ/mol" )
print( "E_t  ", round( e_t, 2 ), ", dE: ", round( act, 2 ), "_kJ/mol"  )
print( "E_p  ", round( e_p, 2 ), "_kJ/mol" )
print( "ifrq ", round( frq, 2 ), "_cm^-1" )
print( "Temp ", round( tem, 2 ), "_K" )
A = e_p - e_r
print( "A    ", round( A, 2 ), "_kJ/mol" )
B = 2.0 * act - A + 2.0 * math.sqrt( act * ( act - A ) )
print( "B    ", round( B, 2 ), "_kJ/mol" )
y = A / B
print( "y    ", round( y, 4 ) )
tet = 83.067 / frq * math.sqrt( B * math.pow( 1.0 - y, 2.0 ) * ( 1.0 - y + 2.0 * y * y ) )
plt.clf()
plt.grid( True )
plt.ylabel( "dE [kJ/mol]" )
plt.xlabel( "reac. coor. [A]" )
x = numpy.linspace( -2, 2, 101 )
z = numpy.exp( pi2 * x / tet )
v = A * z / ( 1 + z ) + B * z / numpy.square( 1 + z )
plt.plot( x, v, '-' )
plt.show()
print( "tet  ", round( tet, 4 ), "_A" )
C = 1.975 / ( tet * tet )
print( "C    ", round( C, 2 ), "_kJ/mol" )
delt = 0.5 * math.sqrt( ( B - C ) / C )
dw = 0.001
w  = 2.0 * act
w  = numpy.linspace( 0, w, int( round( w / dw, 0 ) ) + 1 )
dw = w[1] - w[0]
alph = 0.5 * numpy.sqrt( w / C )
beta = 0.5 * numpy.sqrt( ( w - A ) / C )
apb = numpy.cosh( pi2 * ( alph + beta ) )
G  = ( apb - numpy.cosh( pi2 * ( alph - beta ) ) ) / ( apb + numpy.cosh( pi2 * delt ) )
plt.clf()
plt.grid( True )
plt.ylabel( "G(w)" )
plt.xlabel( "w [kJ/mol]" )
plt.plot( w, G, '-' )
plt.show()
G *= numpy.exp( - cte * w / tem )
kapp = math.exp( cte * act / tem ) * cte / tem * numpy.trapz( G, dx = dw )
print( "\nkappa %20.10le"%( kapp ) )
print( "dEtun ", round( - 8.314 * tem * math.log( kapp ) / 1000., 2 ), "_kJ/mol" )

#print( numpy.trapz( G, dx = dw ) )
#import  qm3.utils
#print( qm3.utils.Simpson( w, G ) )
#import  qm3.utils.interpolation
#obj = qm3.utils.interpolation.cubic_spline( w, G )
#print( qm3.utils.Gauss_Legendre( lambda x: obj.calc( x )[0], w[0], w[-1] ) )
