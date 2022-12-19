#!/usr/bin/env python3
import  sys
import  numpy
import  qm3
import  qm3.utils._dcd
import  matplotlib.pyplot as plt

mol = qm3.molecule()
mol.prmtop_read( open( "../just_protein.prmtop" ) )

sel = mol.labl == "C"
sel = numpy.logical_or( sel, mol.labl == "N" )
sel = numpy.logical_or( sel, mol.labl == "CA" )

sel = mol.anum > 1

dcd = qm3.utils._dcd.dcd()
dcd.open_read( "just_protein.dcd" )
dcd.next( mol )

ref = mol.coor[sel]
ref -= numpy.average( ref, axis = 0 )

out = []
while( dcd.next( mol ) ):
    cur = mol.coor[sel]
    cur -= numpy.average( cur, axis = 0 )
    cov = numpy.dot( cur.T, ref )
    r1, s, r2 = numpy.linalg.svd( cov )
    if( numpy.linalg.det( cov ) < 0 ):
        r2[2,:] *= -1.0
    mod = numpy.dot( cur, numpy.dot( r1, r2 ) )
    out.append( numpy.sqrt( numpy.average( numpy.sum( ( ref - mod ) ** 2, axis = 1 ) ) ) )
dcd.close()
out = numpy.array( out )
plt.clf()
plt.grid( True )
plt.plot( out, '-' )
plt.savefig( "rmsd.pdf" )
plt.show()
