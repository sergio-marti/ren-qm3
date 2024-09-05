#!/usr/bin/env python3
import  sys
import  os
import  numpy
import  qm3.utils.interpolation
import  matplotlib.pyplot as plt
import  zipfile


try:
    skp = int( os.environ["SKIP"] )
except:
    skp = 0

try:
    frz = int( os.environ["ALL"] ) == 1
except:
    frz = False

zzz = zipfile.ZipFile( sys.argv[1], "r" )
dat = []
print( "%-10s%8s%8s%8s%10s"%( "window", "m-2.5s", "m  ", "m+2.5s", "%(x<>m)" ) )
for w in zzz.namelist():
    if( w[0:3] == "dat" ):
        f = zzz.open( w, "r" )
        f.readline()
        x = numpy.loadtxt( f )[skp:]
        f.close()
        m = numpy.mean( x )
        s = numpy.std( x )
        q = max( numpy.sum( x <= m ), numpy.sum( x >= m ) ) / x.shape[0]
        print( "%-10s%8.3lf%8.3lf%8.3lf%10.3lf"%( w, m - 2.5 * s, m, m + 2.5 * s, q ) )
        dat.append( [ m, m - 2.5 * s, m + 2.5 * s, q, w[4:] ] )
dat.sort()
sel = numpy.ones( len( dat ), dtype=numpy.bool_ )
if( not frz ):
    flg = True
    for i in range( 6, -1, -1 ):
        flg = flg and ( dat[i][2] >= dat[i+1][1] ) and ( dat[i][3] <= 0.75 )
        sel[i] = flg
    flg = True
    for i in range( -6, 0, 1 ):
        flg = flg and ( dat[i][1] <= dat[i-1][2] ) and ( dat[i][3] <= 0.75 )
        sel[i] = flg
sel = numpy.flatnonzero( sel )
print( sel )
lst = [ dat[i][-1] for i in sel ]
dat = numpy.array( [ dat[i][0] for i in sel ] )

avr = []
for w in lst:
    f = zzz.open( "geo." + w, "r" )
    avr.append( numpy.mean( numpy.loadtxt( f )[skp:], axis = 0 ) )
    f.close()
avr = numpy.array( avr ).T
print( avr.shape )

zzz.close()

gau = []
for i in range( avr.shape[0] ):
    fit = qm3.utils.interpolation.savitzky_golay( avr[i], 13 )
    obj = qm3.utils.interpolation.gaussian( dat, fit, 0.1 )
    gau.append( [ obj.calc( j )[0] for j in dat ] )

f = open( "geom.clr", "wt" )
for i in range( avr.shape[1] ):
    f.write( "%20.10lf"%( dat[i] ) )
    for j in range( avr.shape[0] ):
        f.write( "%20.10lf"%( gau[j][i] ) )
    f.write( "\n" )
f.close()

plt.clf()
plt.grid( True )
for i in range( avr.shape[0] ):
    plt.plot( dat, avr[i], "." )
for i in range( avr.shape[0] ):
    plt.plot( dat, gau[i], "-", linewidth = 2.0 )
plt.savefig( "geom.pdf" )
