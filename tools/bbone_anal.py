#!/usr/bin/env python3
import  numpy
import  qm3
import  qm3.utils._dcd
import  matplotlib.pyplot as plt
import  matplotlib.backends.backend_pdf
import  sklearn.decomposition

mol = qm3.molecule()
mol.prmtop_read( open( "prmtop" ) )

idx = mol.labl == "CA"
sel = numpy.logical_or( idx, mol.labl == "C" )
sel = numpy.logical_or( sel, mol.labl == "N" )

sel = numpy.flatnonzero( sel )
idx = numpy.flatnonzero( numpy.in1d( sel, numpy.flatnonzero( idx ) ) )

dcd = qm3.utils._dcd.dcd()
dcd.open_read( "dcd" )
dcd.next( mol )

ref = mol.coor[sel]
ref -= numpy.mean( ref, axis = 0 )
siz = numpy.sqrt( ref.shape[0] )

out = []
dif = []
crd = []
while( dcd.next( mol ) ):
    cur = mol.coor[sel]
    cur -= numpy.mean( cur, axis = 0 )
    cov = numpy.dot( cur.T, ref )
    r1, s, r2 = numpy.linalg.svd( cov )
    if( numpy.linalg.det( cov ) < 0 ):
        r2[2,:] *= -1.0
    cur = numpy.dot( cur, numpy.dot( r1, r2 ) )
    crd.append( cur )
    dif.append( cur[idx] - ref[idx] )
    #out.append( numpy.sqrt( numpy.mean( numpy.sum( numpy.square( cur - mod ), axis = 1 ) ) ) )
    out.append( numpy.linalg.norm( cur - mod ) / siz )
dcd.close()

out = numpy.array( out )
numpy.savetxt( "rmsd.dat", out )
plt.clf()
plt.grid( True )
plt.plot( out, '-' )
plt.xlabel( "Time" )
plt.ylabel( "RMSd $(N,C_{\\alpha},C)$" )
plt.tight_layout()
plt.savefig( "rmsd.pdf" )
plt.show()

med = numpy.mean( dif, axis = 0 )
out = numpy.sqrt( numpy.mean( numpy.sum( numpy.square( dif - med ), axis = 2 ), axis = 0 ) )
numpy.savetxt( "rmsf.dat", out )
plt.clf()
plt.grid( True )
plt.plot( out, '-' )
plt.xlabel( "Residue" )
plt.ylabel( "RMSf $(C_{\\alpha})$" )
plt.tight_layout()
plt.savefig( "rmsf.pdf" )
plt.show()

dif = numpy.array( dif )
dr2 = numpy.sqrt( numpy.mean( numpy.sum( numpy.square( dif ), axis = 2 ), axis = 0 ) )
img = numpy.zeros( ( dif.shape[1], dif.shape[1] ) )
with open( "dccm.dat", "wt" ) as f:
    for i in range( dif.shape[1] ):
        for j in range( dif.shape[1] ):
            img[i,j] = numpy.sum( dif[:,i,:] * dif[:,j,:] ) / ( dif.shape[0] * dr2[i] * dr2[j] )
            f.write( "%10d%10d%20.10lf\n"%( i+1, j+1, img[i,j] ) )
        f.write( "\n" )
plt.clf()
plt.grid( True )
plt.imshow( img, cmap = "coolwarm" )
plt.colorbar()
plt.xlabel( "Residue $(C_{\\alpha})$" )
plt.ylabel( "Residue $(C_{\\alpha})$" )
plt.tight_layout()
plt.savefig( "dccm.pdf" )
plt.show()

crd = numpy.array( crd )
crd.shape = ( crd.shape[0], crd.shape[1] * crd.shape[2] )
pca = sklearn.decomposition.PCA( n_components = 2 )
red = pca.fit_transform( crd )
with open( "pca.dat", "wt" ) as f:
    for i in range( crd.shape[0] ):
        f.write( "%12.6lf%12.6lf\n"%( red[i,0], red[i,1] ) )
val, vec = numpy.linalg.eigh( pca.get_covariance() )
print( val[-2:] )
lbl = qm3.data.symbol[mol.anum[sel]]
for k in [1, 2]:
    with open( "mode%d.xyz"%( k ), "wt" ) as f:
        mod = vec[:,-k].reshape( ( lbl.shape[0], 3 ) )
        for i in range( 28 ):
            f.write( "%d\n\n"%( lbl.shape[0] ) )
            # empirical factor... :d
            fac = 8.5 * numpy.sin( numpy.pi * float( i ) / 14 )
            for j in range( lbl.shape[0] ):
                f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( lbl[j],
                    ref[j,0] + fac * mod[j,0],
                    ref[j,1] + fac * mod[j,1],
                    ref[j,2] + fac * mod[j,2] ) )
pdf = matplotlib.backends.backend_pdf.PdfPages( "pca.pdf" )
plt.clf()
plt.grid( True )
plt.xlabel( "Cartesian coordinate PCA" )
plt.ylabel( "Probability" )
x = numpy.linspace( numpy.min( red ), numpy.max( red ), 100 )
y, _ = numpy.histogram( red[:,0], bins = x, density = True ) 
plt.plot( x[:-1], y, '-', label = "PC1" )
y, _ = numpy.histogram( red[:,1], bins = x, density = True ) 
plt.plot( x[:-1], y, '-', label = "PC2" )
plt.legend( loc = "upper right", fontsize = "small" )
plt.tight_layout()
pdf.savefig()
plt.show()
plt.clf()
plt.grid( True )
plt.scatter( red[:,0], red[:,1], marker = "o", c = numpy.arange( crd.shape[0] ) )
plt.xlabel( "PC1" )
plt.ylabel( "PC2" )
plt.title( "Cartesian coordinate PCA" )
cbar = plt.colorbar()
cbar.set_label( "Time" )
plt.tight_layout()
pdf.savefig()
plt.show()
pdf.close()
