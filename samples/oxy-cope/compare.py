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
    neb.append( [] )
    with zz.open( fn, "r" ) as f:
        mol.xyz_read( f )
        neb[-1].append( qm3.utils.distance( mol.coor[bnd[0]], mol.coor[bnd[1]] ) - qm3.utils.distance( mol.coor[bnd[2]], mol.coor[bnd[3]] ) )
        f.seek( 0 )
        f.readline()
        neb[-1].append( float( f.readline().split()[-1] ) )
zz.close()
neb = numpy.array( neb )
neb[:,1] = ( neb[:,1] - neb[0,1] ) / 4.184

pmf = numpy.loadtxt( "pmf.txt" )

pes = numpy.loadtxt( "path.log" )

pmf_s = numpy.loadtxt( "pmf_s.txt" )
geo_s = numpy.loadtxt( "geom.clr" )
#obj   = qm3.utils.interpolation.cubic_spline( geo_s[:,0], geo_s[:,1] - geo_s[:,2] )
obj   = qm3.utils.interpolation.gaussian( geo_s[:,0], geo_s[:,1] - geo_s[:,2], 0.1 )
crd_s = numpy.array( [ obj.calc( x )[0] for x in pmf_s[:,0] ] )

str_x = numpy.loadtxt( "string.cvar" )
str_f = numpy.loadtxt( "string.mfep" ) / 4.184

plt.clf()
plt.grid( True )
plt.plot( pes[:,0] - pes[:,1], pes[:,2] - pes[0,2], '.-', label = "PES" )
plt.plot( neb[:,0], neb[:,1], '.-', label = "NEB" )
plt.plot( pmf[:,0], pmf[:,2], '.-', label = "PMF" )
plt.plot( str_x[:,0] - str_x[:,1], str_f, '.-', label = "String" )
plt.plot( crd_s, pmf_s[:,1], '.-', label = "PMF_s" )
plt.legend( loc = "upper right", fontsize = "small" )
plt.tight_layout()
plt.savefig( "compare.pdf" )
