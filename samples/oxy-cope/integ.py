import  glob
import  numpy
import  matplotlib.pyplot as plt
import  matplotlib.backends.backend_pdf
import  qm3.utils.pmf

pdf = matplotlib.backends.backend_pdf.PdfPages( "plt.pdf" )

lst = list( glob.glob( "dat.??" ) )

skp = 800

#crd, pmf = qm3.utils.pmf.wham( [ open( f ) for f in lst ], nskip = skp )
#err = numpy.zeros( len( crd ) )

crd, pmf, err = qm3.utils.pmf.umbint( [ open( f ) for f in lst ], nskip = skp )
pmf -= pmf[1]
pmf /= 4.184
err /= 4.184

with open( "pmf", "wt" ) as f:
    for i in range( 1, len( crd ) ):
        f.write( "%20.10lf%20.10lf%20.10lf\n"%( crd[i], pmf[i], err[i] ) )

plt.clf()
plt.grid( True )
plt.plot( crd[1:], pmf[1:], '-o' )
pdf.savefig()

plt.clf()
plt.grid( True )
for dat in lst:
    with open( dat, "rt" ) as f:
        for i in range( skp + 1 ):
            f.readline()
        plt.plot( [ float( l.strip() ) for l in f ], '-' )
pdf.savefig()

pdf.close()
