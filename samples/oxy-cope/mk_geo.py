#!/usr/bin/env python3
import  sys
import	math
import  numpy
import  zipfile
import  qm3.utils.interpolation
import  matplotlib.pyplot as plt


zzz = zipfile.ZipFile( sys.argv[1], "r" )
lst = sorted( [ int( i[4:] ) for i in zzz.namelist() if( i[0:3] == "geo" ) ] )
num = len( lst )

with zzz.open( "geo.%02d"%( lst[0] ), "r" ) as f:
    siz = len( f.readline().split() )
print( siz, num )

avr = [ [ .0 for j in range( num ) ] for i in range( siz ) ]
rms = [ [ .0 for j in range( num ) ] for i in range( siz ) ]
npt = [ [ .0 for j in range( num ) ] for i in range( siz ) ]

for i in range( num ):
    f = zzz.open( "geo.%02d"%( lst[i] ), "r" )
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
    f = zzz.open( "dat.%02d"%( i ), "r" )
    f.readline()
    m = [ float( l.strip() ) for l in f ][skp-1:]
    f.close()
    x.append( sum( m ) / len( m ) )
x = numpy.array( x )

zzz.close()

fit = []
gau = []
for i in range( siz ):
    fit.append( qm3.utils.interpolation.savitzky_golay( avr[i], 13 ) )
    #obj = qm3.utils.interpolation.gaussian( x, fit[i], 0.1 )
    obj = qm3.utils.interpolation.cubic_spline( x, fit[i] )
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
for i in range( siz ):
    plt.plot( x, gau[i], "-", linewidth = 2.0 )
plt.savefig( "geom.pdf" )
