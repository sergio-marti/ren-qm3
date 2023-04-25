#!/usr/bin/env python3
import  qm3
import  numpy
import  qm3.utils._cions

m = qm3.molecule()
m.xyz_read( open( "../mixed_xtb.xyz" ) )
m.guess_atomic_numbers()
with open( "../mixed_xtb.chrg" ) as f:
    m.chrg = numpy.array( [ float( i ) for i in f.read().split() ] )
print( m.natm, len( m.chrg ), m.chrg.sum() )
crd = qm3.utils._cions.counter_ions( m, 4, -1, 0.5, 11, 6.5, 8 )
with open( "neutral.xyz", "wt" ) as f:
    f.write( "%d\n\n"%( m.natm + 4 ) )
    for i in range( m.natm ):
        f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( m.labl[i], m.coor[i,0], m.coor[i,1], m.coor[i,2] ) )
    for i in range( 4 ):
        f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( "Cl", crd[i,0], crd[i,1], crd[i,2] ) )
