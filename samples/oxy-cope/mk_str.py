#!/usr/bin/env python3
import  numpy
import  io
import  qm3
import  qm3.engines.mmres
import  matplotlib.pyplot as plt
import  zipfile

zzz = zipfile.ZipFile( "neb.zip", "r" )

lst = [ s for s in sorted( zzz.namelist() ) if s[0:4] == "node" ]

mol = qm3.molecule()
mol.psf_read( open( "oxy-cope.psf", "rt" ) )
mol.xyz_read( zzz.open( lst[0], "r" ), replace = True )

f = io.StringIO( """
A/1/C2      A/1/C3
A/1/C5      A/1/C6
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

with open( "pmf_s.cnf", "wt" ) as f_cnf:
    for i,j in bnd:
        f_cnf.write( "%8d%8d\n"%( i, j ) )

obj = qm3.engines.mmres.colvar_s( mol, "pmf_s.cnf" )
obj.append( mol )
for w in lst[1:]:
    mol.xyz_read( zzz.open( w, "r" ), replace = True )
    obj.append( mol )

zzz.close()

delz, dels, arcl = obj.define( "pmf_s.str", redistribute = True )

with open( "pmf_s.delz", "wt" ) as f:
    f.write( "%.6lf\n"%( delz ) )

with open( "pmf_s.dels", "wt" ) as f:
    f.write( "%.6lf\n"%( dels ) )
