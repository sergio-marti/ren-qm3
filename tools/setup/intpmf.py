#!/usr/bin/env python3
import  sys
import  os
import  numpy
import  matplotlib.pyplot as plt
import  matplotlib.backends.backend_pdf
import  qm3.utils.pmf
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
tmp = []
print( "%-10s%8s%8s%8s%10s"%( "window", "m-2.5s", "m  ", "m+2.5s", "%(x<>m)" ) )
for w in zzz.namelist():
    if( w[0:3] == "dat" ):
        f = zzz.open( w, "r" )
        f.readline()
        x = numpy.loadtxt( f )[skp:]
        m = numpy.mean( x )
        s = numpy.std( x )
        q = max( numpy.sum( x <= m ), numpy.sum( x >= m ) ) / x.shape[0]
        print( "%-10s%8.3lf%8.3lf%8.3lf%10.3lf"%( w, m - 2.5 * s, m, m + 2.5 * s, q ) )
        tmp.append( [ m, m - 2.5 * s, m + 2.5 * s, q, f ] )
tmp.sort()
mny = tmp[0][0]
mxy = tmp[0][0]
sel = numpy.ones( len( tmp ), dtype=numpy.bool_ )
if( not frz ):
    flg = True
    for i in range( 6, -1, -1 ):
        flg = flg and ( tmp[i][2] >= tmp[i+1][1] ) and ( tmp[i][3] <= 0.75 )
        sel[i] = flg
    flg = True
    for i in range( -6, 0, 1 ):
        flg = flg and ( tmp[i][1] <= tmp[i-1][2] ) and ( tmp[i][3] <= 0.75 )
        sel[i] = flg
print( numpy.flatnonzero( sel ) )
lst = [ tmp[i][-1] for i in numpy.flatnonzero( sel ) ]

pdf = matplotlib.backends.backend_pdf.PdfPages( "plt.pdf" )

for f in lst:
    f.seek( 0 )

umb_c, umb_f, umb_e = qm3.utils.pmf.umbint( lst, nskip = skp )
t = min( umb_f[0:len(umb_c)//2] )
for i in range( len( umb_c ) ):
    umb_f[i] = ( umb_f[i] - t ) / 4.184

for f in lst:
    f.seek( 0 )

pmf_c, pmf_f = qm3.utils.pmf.wham( lst, maxit = 20000, nskip = skp )
t = min( pmf_f[0:len(pmf_c)//2] )
f = open( "ene", "wt" )
for i in range( len( pmf_c ) ):
    pmf_f[i] = ( pmf_f[i] - t ) / 4.184
    f.write( "%20.10lf%20.10lf%20.10lf\n"%( pmf_c[i], pmf_f[i], umb_f[i] ) )
f.close()

plt.clf()
plt.grid( True )
plt.plot( umb_c, umb_f, '-' )
plt.xlabel( "reaction coordinate (Å)" )
plt.ylabel( "∆PMF (kcal/mol)" )
plt.plot( pmf_c, pmf_f, '.-', linewidth = 2.0 )
plt.tight_layout()
pdf.savefig()
plt.show()

plt.clf()
plt.grid( True )
dsp = 0
for f in lst:
    f.seek( 0 )
    f.readline()
    x = numpy.loadtxt( f )[skp:]
    dsp = max( dsp, numpy.std( x ) )
    mny = min( mny, numpy.min( x ) )
    mxy = max( mny, numpy.max( x ) )
    plt.plot( x, '-' )
plt.ylim( ( mny - dsp, mxy + dsp ) )
plt.tight_layout()
pdf.savefig()

sel = numpy.flatnonzero( numpy.logical_not( sel ) )
if( sel.shape[0] > 0 ):
    plt.clf()
    plt.grid( True )
    dsp = 0
    for i in sel:
        tmp[i][-1].seek( 0 )
        tmp[i][-1].readline()
        x = numpy.loadtxt( tmp[i][-1] )[skp:]
        dsp = max( dsp, numpy.std( x ) )
        mny = min( mny, numpy.min( x ) )
        mxy = max( mny, numpy.max( x ) )
        plt.plot( x, '-' )
    plt.ylim( ( mny - dsp, mxy + dsp ) )
    plt.tight_layout()
    pdf.savefig()

zzz.close()

pdf.close()
