#!/usr/bin/env python3
import  numpy
import  io
import  qm3
import  qm3.data
import  qm3.utils
import  qm3.utils.interpolation
import  qm3.engines.mmres
import  matplotlib.pyplot as plt
import  zipfile


zzz = zipfile.ZipFile( "neb.zip", "r" )
lst = [ s for s in sorted( zzz.namelist() ) if s[0:4] == "node" ]

#zzz = zipfile.ZipFile( "otfs.zip", "r" )
#lst = [ s for s in sorted( zzz.namelist() ) if s[0:4] == "node" ]

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop", "rt" ) )
mol.xyz_read( zzz.open( lst[0], "r" ), replace = True )

f = io.StringIO( """
A/206/OG    A/609/C6
A/206/OG    A/206/HG
A/206/HG    A/312/NE2
A/206/HG    A/609/O
A/609/C6    A/609/O
""" )

bnd = []
for l in f:
    t = l.split()
    if( len( t ) == 2 ):
        a = []
        for b in t:
            _s, _r, _l = b.split( "/" )
            a.append( mol.indx[_s][int(_r)][_l] )
        print( t, a )
        bnd.append( a )

nwin = len( lst )
ncrd = len( bnd )
rcrd = numpy.zeros( ( nwin, ncrd ) )
for i in range( nwin ):
    mol.xyz_read( zzz.open( lst[i], "r" ), replace = True )
    for j in range( ncrd ):
        rcrd[i,j] = qm3.utils.distance( mol.coor[bnd[j][0]], mol.coor[bnd[j][1]] )

with open( "pmf_s.cnf", "wt" ) as f_cnf:
    f_cnf.write( "%d %d\n"%( ncrd, nwin ) )
    for j in range( ncrd ):
        f_cnf.write( "%8d%8d\n"%( bnd[j][0], bnd[j][1] ) )

f_str = io.StringIO()
for i in range( nwin ):
    for j in range( ncrd ):
        f_str.write( "%12.4lf"%( rcrd[i,j] ) )
    f_str.write( "\n" )

f_str.seek( 0 )
obj = qm3.engines.mmres.colvar_s( mol, .0, .0, open( "pmf_s.cnf" ), f_str, None )

with open( "pmf_s.met", "wt" ) as f_met:
    for i in range( nwin ):
        mol.xyz_read( zzz.open( lst[i], "r" ), replace = True )
        f_met.write( "".join( [ "%8.3lf"%( i ) for i in obj.metrics( mol ).ravel().tolist() ] ) + "\n" )

zzz.close()

f_str.seek( 0 )
obj = qm3.engines.mmres.colvar_s( mol, .0, .0, open( "pmf_s.cnf" ), f_str, open( "pmf_s.met" ) )

plt.clf()
plt.grid( True )
plt.plot( obj.arcl[1:], '-o' )
plt.show()

arcl = numpy.cumsum( obj.arcl )
equi = numpy.array( [ arcl[-1] / ( nwin - 1.0 ) * i for i in range( nwin ) ] )
fcrd = numpy.zeros( ( nwin, ncrd ) )
print( arcl )
plt.clf()
plt.grid( True )
for j in range( ncrd ):
    #inte = qm3.utils.interpolation.hermite_spline( arcl, rcrd[:,j], "akima" )
    inte = qm3.utils.interpolation.cubic_spline( arcl, rcrd[:,j] )
    inte = qm3.utils.interpolation.gaussian( arcl, rcrd[:,j], 0.2 )
    fcrd[:,j] = numpy.array( [ inte.calc( x )[0] for x in equi ] )
    plt.plot( rcrd[:,j], '-' )
    plt.plot( fcrd[:,j], 'o' )
plt.savefig( "setup.pdf" )
plt.show()

with open( "pmf_s.str", "wt" ) as f_str:
    for i in range( nwin ):
        for j in range( ncrd ):
            f_str.write( "%12.4lf"%( fcrd[i,j] ) )
        f_str.write( "\n" )

obj = qm3.engines.mmres.colvar_s( mol, .0, .0,
        open( "pmf_s.cnf" ), open( "pmf_s.str" ), open( "pmf_s.met" ) )

with open( "pmf_s.delz", "wt" ) as f:
    f.write( "%.6lf\n"%( obj.delz ) )

plt.clf()
plt.grid( True )
plt.plot( obj.arcl[1:], '-o' )
plt.show()
