#!/usr/bin/env python3
import  numpy
import  qm3
import  define_QM
import  pickle


mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop") )
mol.xyz_read( open( "guess" ) )
with open( "namd_npt.xsc" ) as f:
    l = f.readline()
    while( l[0] == "#" ):
        l = f.readline()
    tmp = [ float( s ) for s in l.split() ]
    mol.boxl = numpy.array( [ tmp[1], tmp[5], tmp[9] ] )


con = [ [] for i in range( mol.natm ) ]
bnd = define_QM.parse_prmtop( "complex.prmtop" )
for i,j in bnd:
    con[i].append( j )
    con[j].append( i )

sqm = list( mol.indx["A"][609].values() )
sqm += define_QM.get_atoms( con,  mol.indx["A"][312]["CB"], [ mol.indx["A"][312]["CA"] ] )
sqm += define_QM.get_atoms( con,  mol.indx["A"][206]["CB"], [ mol.indx["A"][206]["C"], mol.indx["A"][205]["CA"] ] )
sqm.sort()
with open( "sele_QM.pk", "wb" ) as f:
    pickle.dump( sqm, f )

sla = []
for i,j in bnd:
    if( i in sqm and not j in sqm ):
        sla.append( [ i, j ] )
    elif( j in sqm and not i in sqm ):
        sla.append( [ j, i ] )
with open( "sele_LA.pk", "wb" ) as f:
    pickle.dump( sla, f )

tmp = numpy.zeros( mol.natm, dtype=numpy.bool_ )
tmp[sqm] = True
tmp = numpy.argwhere( mol.sph_sel( tmp, 20 ) ).ravel().tolist()
smm = list( sorted( set( tmp ).difference( set( sqm + sum( sla, [] ) ) ) ) )
with open( "sele_MM.pk", "wb" ) as f:
    pickle.dump( smm, f )

with open( "borra", "wt" ) as f:
    tmp = numpy.zeros( mol.natm, dtype=numpy.bool_ )
    tmp[sqm] = True
    mol.xyz_write( f, sele = tmp )
    tmp = numpy.zeros( mol.natm, dtype=numpy.bool_ )
    tmp[[ j for i,j in sla]] = True
    mol.xyz_write( f, sele = tmp )
    tmp = numpy.zeros( mol.natm, dtype=numpy.bool_ )
    tmp[smm] = True
    mol.xyz_write( f, sele = tmp )
