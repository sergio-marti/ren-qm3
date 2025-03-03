#!/usr/bin/env python3
import  numpy
import  matplotlib.pyplot as plt
import  qm3
import  qm3.utils
import  qm3.utils.interpolation
import  zipfile

mol = qm3.molecule()
mol.psf_read( open( "oxy-cope.psf" ) )
bnd = [ mol.indx["A"][1]["C2"], mol.indx["A"][1]["C3"], mol.indx["A"][1]["C5"], mol.indx["A"][1]["C6"] ]
neb = []
zz = zipfile.ZipFile( "neb.zip", "r" )
for fn in sorted( zz.namelist() ):
    with zz.open( fn, "r" ) as f:
        mol.xyz_read( f )
        neb.append( [ qm3.utils.distance( mol.coor[bnd[0]], mol.coor[bnd[1]] ), qm3.utils.distance( mol.coor[bnd[2]], mol.coor[bnd[3]] ) ] )
zz.close()
neb = numpy.array( neb )
print( neb.shape )

pmf = []
zz = zipfile.ZipFile( "pmf.zip", "r" )
for fn in sorted( zz.namelist() ):
    if( fn[0:3] == "geo" ):
        with zz.open( fn, "r" ) as f:
            pmf.append( numpy.mean( numpy.loadtxt( f ), axis = 0 ) )
zz.close()
pmf = numpy.array( pmf )
print( pmf.shape )

pes = numpy.loadtxt( "path.log" )

crd_s = numpy.loadtxt( "geom.clr" )

plt.clf()
plt.grid( True )
plt.plot( pes[:,0] - pes[:,1], pes[:,0], '.-', label = "PES [d1]" )
plt.plot( pes[:,0] - pes[:,1], pes[:,1], '.-', label = "PES [d2]" )

plt.plot( neb[:,0] - neb[:,1], neb[:,0], '.-', label = "NEB [d1]" )
plt.plot( neb[:,0] - neb[:,1], neb[:,1], '.-', label = "NEB [d2]" )

plt.plot( pmf[:,0] - pmf[:,1], pmf[:,0], '.-', label = "PMF [d1]" )
plt.plot( pmf[:,0] - pmf[:,1], pmf[:,1], '.-', label = "PMF [d2]" )

str_x = numpy.loadtxt( "1n7_3k/string.cvar" )
plt.plot( str_x[:,0] - str_x[:,1], str_x[:,0], '.-', label = "String [d1]" )
plt.plot( str_x[:,0] - str_x[:,1], str_x[:,1], '.-', label = "String [d2]" )

str_x = numpy.loadtxt( "from_pes/string.cvar" )
plt.plot( str_x[:,0] - str_x[:,1], str_x[:,0], '.-', label = "String:PES [d1]" )
plt.plot( str_x[:,0] - str_x[:,1], str_x[:,1], '.-', label = "String:PES [d2]" )

plt.plot( crd_s[1:,1] - crd_s[1:,2], crd_s[1:,1], '.-', label = "PMF_s [d1]" )
plt.plot( crd_s[1:,1] - crd_s[1:,2], crd_s[1:,2], '.-', label = "PMF_s [d2]" )

plt.legend( loc = "upper right", fontsize = "small" )
plt.tight_layout()
plt.savefig( "compare_dst.pdf" )
