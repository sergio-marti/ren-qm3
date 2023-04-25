#!/usr/bin/env python3
import sys
import math
import matplotlib.pyplot as pyplot
import matplotlib.backends.backend_pdf
import qm3.utils.pmf


def plot_data( data, dsigma = 2.0, skip = 0 ):
    __mm = []
    __ss = []
    __dd = []
    __Mx = 0
    for fn in data:
        f = open( fn, "rt" )
#        t = [ float( i ) for i in f.readline().strip().split() ]
        f.readline()
        n = 0.0
        m = 0.0
        s = 0.0
        __dd.append( [] )
        c = 0
        for l in f:
            if( c >= skip ):
                t = float( l.strip().split()[0] )
                m += t
                s += t * t
                n += 1.0
                __dd[-1].append( t )
            c += 1
        f.close()
        m /= n
        __mm.append( m )
        __ss.append( math.sqrt( math.fabs( s / n - m * m ) ) )
        __Mx = max( __Mx, n )
    t = sorted( __mm )
    __mw = __mm.index( t[0]  )
    __Mw = __mm.index( t[-1] )
    pyplot.grid( True )
    pyplot.xlim( 0.0, __Mx )
    pyplot.ylim( __mm[__mw] - 2.0 * __ss[__mw], __mm[__Mw] + 2.0 * __ss[__Mw] )
    for i in range( len( __mm ) ):
        x = [ float( j ) for j in range( len( __dd[i] ) ) ]
        pyplot.plot( x, __dd[i] )
    pdf.savefig()
    pyplot.clf()
    pyplot.xlim( __mm[__mw] - 2.0 * __ss[__mw], __mm[__Mw] + 2.0 * __ss[__Mw] )
    f = 1.0 / math.sqrt( 2.0 * math.pi )
    for i in range( len( __mm ) ):
        nx = 100
        mx = __mm[i] - dsigma * __ss[i]
        dx = 2.0 * ( dsigma * __ss[i] ) / float( nx )
        x = [ mx + float( j ) * dx for j in range( nx + 1 ) ]
        y = [ f / __ss[i] * math.exp( - 0.5 * math.pow( ( x[j] - __mm[i] ) / __ss[i], 2.0 ) ) for j in range( nx + 1 ) ]
        pyplot.plot( x, y )
    pdf.savefig()


try:
    SKP = int( sys.argv[1] )
except:
    SKP = 0

f = open( "range", "rt" )
lst = eval( f.readline() )
f.close()
#lst = list( range( 2, 59 ) )

lst = [ "dat.%02d"%( i ) for i in lst ]

pdf = matplotlib.backends.backend_pdf.PdfPages( "plt.pdf" )

umb_c, umb_f, umb_e = qm3.utils.pmf.umbint( lst, SKP )
t = min( umb_f[0:len(umb_c)//2] )
for i in range( len( umb_c ) ):
    umb_f[i] = ( umb_f[i] - t ) / 4.184

pmf_c, pmf_f = qm3.utils.pmf.wham( lst, SKP, maxit = 20000 )
t = min( pmf_f[0:len(pmf_c)//2] )
f = open( "ene", "wt" )
for i in range( len( pmf_c ) ):
    pmf_f[i] = ( pmf_f[i] - t ) / 4.184
    f.write( "%20.10lf%20.10lf%20.10lf\n"%( pmf_c[i], pmf_f[i], umb_f[i] ) )
f.close()

pyplot.clf()
pyplot.grid( True )
pyplot.plot( umb_c, umb_f, '-' )
pyplot.plot( pmf_c, pmf_f, '-' )
pdf.savefig()
pyplot.show()

plot_data( lst, skip = SKP )
pdf.close()
