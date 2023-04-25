#!/usr/bin/env python3
import  sys
import  math


def stats( v ):
    n = float( len( v ) )
    m = sum( v ) / n
    s = math.sqrt( sum( [ (i-m)*(i-m) for i in v ] ) / float( n - 1.0 ) )
    return( m, s )

skp = 1
try:
    skp += int( sys.argv[1] )
except:
    pass

avr = []
for I in range( 1, 59 ):
    f = open( "dat.%02d"%( I ), "rt" )
    f.readline()
    m, r = stats( [ float( l.strip() ) for l in f ][skp-1:] )
    f.close()
    avr.append( ( m, r, I ) )

avr.sort()
sel = []
for i in range( len( avr ) - 1 ):
    print( "%4d%12.4lf%12.4lf"%( avr[i][2], avr[i][0], avr[i+1][0] - avr[i][0] ) )
    if( avr[i+1][0] - avr[i][0] > 0.01 ):
        sel.append( avr[i][2] )
print( sel )
f = open( "range", "wt" )
f.write( str( sel ) + "\n" )
f.close()
