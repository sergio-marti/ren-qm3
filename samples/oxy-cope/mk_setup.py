#!/usr/bin/env python3
import  glob
import  numpy
import  io
import  qm3
import  qm3.data
import  qm3.utils
import  qm3.utils.interpolation
import  qm3.engines.mmres
import  matplotlib.pyplot as plt

lst = sorted( glob.glob( "node.??" ) )
mol = qm3.molecule()
mol.pdb_read( open( lst[0] ) )
mol.psf_read( open( "oxy-cope.psf" ) )

bnd = ( ( mol.indx["A"][1]["C2"], mol.indx["A"][1]["C3"] ),
        ( mol.indx["A"][1]["C5"], mol.indx["A"][1]["C6"] ) )

nwin = len( lst )
ncrd = len( bnd )
rcrd = numpy.zeros( ( nwin, ncrd ) )
tmp = qm3.molecule()
for i in range( nwin ):
    tmp.pdb_read( open( lst[i] ) )
    for j in range( ncrd ):
        rcrd[i,j] = qm3.utils.distance( tmp.coor[bnd[j][0]], tmp.coor[bnd[j][1]] )

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
        tmp.pdb_read( open( lst[i] ) )
        tmp.mass = mol.mass
        f_met.write( "".join( [ "%8.3lf"%( i ) for i in obj.metrics( tmp ).ravel().tolist() ] ) + "\n" )

f_str.seek( 0 )
obj = qm3.engines.mmres.colvar_s( mol, .0, .0, open( "pmf_s.cnf" ), f_str, open( "pmf_s.met" ) )

plt.clf()
plt.grid( True )
plt.plot( obj.arcl[1:], '-o' )
plt.show()

arcl = numpy.cumsum( obj.arcl )
equi = numpy.array( [ arcl[-1] / ( nwin - 1.0 ) * i for i in range( nwin ) ] )
fcrd = numpy.zeros( ( nwin, ncrd ) )
plt.clf()
plt.grid( True )
for j in range( ncrd ):
    inte = qm3.utils.interpolation.hermite_spline( arcl, rcrd[:,j], "akima" )
    fcrd[:,j] = numpy.array( [ inte.calc( x )[0] for x in equi ] )
    plt.plot( rcrd[:,j], '-' )
    plt.plot( fcrd[:,j], 'o' )
plt.show()

with open( "pmf_s.str", "wt" ) as f_str:
    for i in range( nwin ):
        for j in range( ncrd ):
            f_str.write( "%12.4lf"%( fcrd[i,j] ) )
        f_str.write( "\n" )

obj = qm3.engines.mmres.colvar_s( mol, .0, .0,
        open( "pmf_s.cnf" ), open( "pmf_s.str" ), open( "pmf_s.met" ) )

plt.clf()
plt.grid( True )
plt.plot( obj.arcl[1:], '-o' )
plt.show()
