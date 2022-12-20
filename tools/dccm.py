#!/usr/bin/env python3
import  numpy
import  qm3
import  qm3.utils._dcd
import  matplotlib.pyplot as plt

mol = qm3.molecule()
mol.prmtop_read( open( "prmtop" ) )

idx = mol.labl == "CA"
sel = numpy.logical_or( idx, mol.labl == "C" )
sel = numpy.logical_or( sel, mol.labl == "N" )

#sel = mol.anum > 1

sel = numpy.flatnonzero( sel )
idx = numpy.flatnonzero( numpy.in1d( sel, numpy.flatnonzero( idx ) ) )

dcd = qm3.utils._dcd.dcd()
dcd.open_read( "dcd" )
dcd.next( mol )

ref = mol.coor[sel]
ref -= numpy.average( ref, axis = 0 )

dri = []
while( dcd.next( mol ) ):
    cur = mol.coor[sel]
    cur -= numpy.average( cur, axis = 0 )
    cov = numpy.dot( cur.T, ref )
    r1, s, r2 = numpy.linalg.svd( cov )
    if( numpy.linalg.det( cov ) < 0 ):
        r2[2,:] *= -1.0
    mod = numpy.dot( cur, numpy.dot( r1, r2 ) )
    dri.append( mod[idx] - ref[idx] )
dcd.close()
dri = numpy.array( dri )
print( dri.shape )
dr2 = numpy.sqrt( numpy.mean( numpy.sum( numpy.square( dri ), axis = 2 ), axis = 0 ) )
print( dr2.shape )
img = numpy.zeros( ( dri.shape[1], dri.shape[1] ) )
with open( "dccm.dat", "wt" ) as f:
    for i in range( dri.shape[1] ):
        for j in range( dri.shape[1] ):
            img[i,j] = numpy.sum( dri[:,i,:] * dri[:,j,:] ) / ( dri.shape[0] * dr2[i] * dr2[j] )
            f.write( "%10d%10d%20.10lf\n"%( i+1, j+1, img[i,j] ) )
        f.write( "\n" )

plt.clf()
plt.grid( True )
plt.imshow( img, cmap = "coolwarm" )
plt.colorbar()
plt.tight_layout()
plt.savefig( "dccm.pdf" )
plt.show()
