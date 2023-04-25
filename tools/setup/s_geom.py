#!/usr/bin/env python3
import	math
import  numpy
import  qm3.utils.interpolation
import  matplotlib.pyplot as plt

leg = []
#leg = [ "$S_\\gamma-H_\\gamma$", "$H_\\gamma-N_\\epsilon$", "$S_\\gamma-C_\\beta$", "$H_\\gamma-O^{{}*{}}$", "$H^{{}*{}}-O^{{}*{}}$", "$H^{{}*{}}-O$" ]

f = open( "pmf_s.cnf", "rt" )
f.readline()
siz = len( f.readlines() )
f.close()

f = open( "range", "rt" )
lst = eval( f.readline() )
f.close()
num = len( lst )
print( siz, num )

avr = [ [ .0 for j in range( num ) ] for i in range( siz ) ]
rms = [ [ .0 for j in range( num ) ] for i in range( siz ) ]
npt = [ [ .0 for j in range( num ) ] for i in range( siz ) ]

for i in range( num ):
    f = open( "geo.%02d"%( lst[i] ), "rt" )
    for l in f:
        t = [ float( j ) for j in l.strip().split() ]
        for j in range( siz ):
            avr[j][i] += t[j]
            rms[j][i] += t[j] * t[j]
            npt[j][i] += 1.0
    f.close()

f = open( "geom.avr", "wt" )
for i in range( num ):
    for j in range( siz ):
        avr[j][i] /= npt[j][i]
        f.write( "%20.10lf"%( avr[j][i] ) )
    f.write( "\n" )
f.close()

skp = 1000
x = []
for i in lst:
    f = open( "dat.%02d"%( i ), "rt" )
    f.readline()
    m = [ float( l.strip() ) for l in f ][skp-1:]
    f.close()
    x.append( sum( m ) / len( m ) )
x = numpy.array( x )
plt.clf()
plt.grid( True )
plt.plot( numpy.diff( x ), '-o' )
plt.show()

fit = []
gau = []
for i in range( siz ):
    fit.append( qm3.utils.interpolation.savitzky_golay( avr[i] ) )
    obj = qm3.utils.interpolation.gaussian( x, fit[i], 0.1 )
    gau.append( [ obj.calc( j )[0] for j in x ] )

f = open( "geom.clr", "wt" )
for i in range( num ):
    f.write( "%20.10lf"%( x[i] ) )
    for j in range( siz ):
        f.write( "%20.10lf"%( gau[j][i] ) )
    f.write( "\n" )
f.close()

plt.clf()
plt.grid( True )
for i in range( siz ):
    plt.plot( x, avr[i], "." )
if( len( leg ) == siz ):
    for i in range( siz ):
        plt.plot( x, gau[i], "-", linewidth = 2.0, label = leg[i] )
    plt.legend( loc = "upper right", fontsize = "medium" )
else:
    for i in range( siz ):
        plt.plot( x, gau[i], "-", linewidth = 2.0 )

plt.savefig( "geom.pdf" )
