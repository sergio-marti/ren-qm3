#!/usr/bin/env python3
import  os
import  numpy
import  qm3
import  qm3.data
import  qm3.utils

vol = 60

if( not os.path.isfile( "box.pdb" ) ):
    d = qm3.data.water_density()
    m = qm3.data.mass[8] + 2 * qm3.data.mass[1]
    n = int( round( d * qm3.data.NA * ( vol * vol * vol * 1.e-24 ) / m, 0 ) )
    with open( "wat", "wt" ) as f:
        f.write( """
HETATM    1  OH2 HOH     1      10.203   7.604  12.673
HETATM    2  H1  HOH     1       9.626   6.787  12.673
HETATM    3  H2  HOH     1       9.626   8.420  12.673
CONECT    1    2
CONECT    1    3
CONECT    1    2    3
END
""" )
    with open( "inp", "wt" ) as f:
        f.write( """tolerance 2.0
output box.pdb
filetype pdb
structure wat
  number %d
  inside cube 0.5 0.5 0.5 %.1lf
end structure
"""%( n, vol - 0.5 ) )
    os.system( "./packmol < inp" )
    print( n )

m = qm3.molecule()
m.xyz_read( open( "neutral.xyz" ) )
m.guess_atomic_numbers()
c = numpy.mean( m.coor, axis = 0 )
m.coor -= c

s = qm3.molecule()
s.pdb_read( open( "box.pdb" ) )
c = numpy.mean( s.coor, axis = 0 )
s.coor -= c

c = 2.8 * 2.8
x = numpy.ones( s.natm, dtype=numpy.bool_ )
for j in range( m.natm ):
    if( m.anum[j] > 1 ):
        for i in range( 0, s.natm, 3 ):
            if( qm3.utils.distanceSQ( m.coor[j,:], s.coor[i,:] ) <= c ):
                x[i:i+3] = False
print( s.natm / 3, x.sum() / 3, ( s.natm - x.sum() ) / 3 )

w = s.copy( x )
with open( "solv.xyz", "wt" ) as f:
    f.write( "%d\n\n"%( m.natm + w.natm ) )
    for i in range( m.natm ):
        f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( m.labl[i], m.coor[i,0], m.coor[i,1], m.coor[i,2] ) )
    for i in range( w.natm ):
        f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( w.labl[i][0], w.coor[i,0], w.coor[i,1], w.coor[i,2] ) )
