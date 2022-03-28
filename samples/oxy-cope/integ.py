import  numpy
import  matplotlib.pyplot as plt
import  matplotlib.backends.backend_pdf
import  qm3.utils.pmf

pdf = matplotlib.backends.backend_pdf.PdfPages( "plt.pdf" )

lst = [ "dat.%02d"%( i ) for i in range( 2, 40 ) ]

skp = 1000

#crd, pmf = qm3.utils.pmf.wham( lst, nskip = skp )
#err = numpy.zeros( len( crd ) )

crd, pmf, err = qm3.utils.pmf.umbint( lst, nskip = skp )
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
plt.show()

plt.clf()
plt.grid( True )
for dat in lst:
    with open( dat, "rt" ) as f:
        for i in range( skp + 1 ):
            f.readline()
        plt.plot( [ float( l.strip() ) for l in f ], '-' )
pdf.savefig()
plt.show()

pdf.close()
