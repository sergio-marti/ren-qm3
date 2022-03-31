#!/usr/bin/env python3
import  sys
import  numpy
import  qm3

mol = qm3.molecule()
mol.pdb_read( sys.argv[1] )

sel = mol.labl == "C"
sel = numpy.logical_or( mol.labl == "N" )
sel = numpy.logical_or( mol.labl == "CA" )

ref = mol.coor[sel]
ref -= numpy.average( ref, axis = 0 )

dcd = qm3.fio.dcd.dcd()
dcd.open_read( sys.argv[2] )

while( dcd.next( mol ) ):
    cur = mol.coor[sel]
    cur -= numpy.average( cur, axis = 0 )
    cov = numpy.dot( cur.T, ref )
    r1, s, r2 = numpy.linalg.svd( cov )
    if( numpy.linalg.det( cov ) < 0 ):
        r2[2,:] *= -1.0
    mod = numpy.dot( cur, numpy.dot( r1, r2 ) )
    print( numpy.sqrt( numpy.average( numpy.sum( ( ref - mod ) ** 2, axis = 1 ) ) ) )

dcd.close()
