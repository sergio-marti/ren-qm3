#!/usr/bin/env python3
import	matplotlib.pyplot as plt

crd = []
pc1 = []
pc2 = []
pc3 = []
with open( "hist.dat", "rt" ) as f:
    f.readline()
    for l in f:
        t = [ float( i ) for i in l.strip().split() ]
        crd.append( t[0] )
        pc1.append( t[1] )
        pc2.append( t[2] )
        pc3.append( t[3] )

plt.clf
plt.grid( True )
plt.xlabel( "Cartesian coordinate PCA [Å]" )
plt.ylabel( "Probability" )
plt.plot( crd, pc1, '-' )
plt.plot( crd, pc2, '-' )
plt.plot( crd, pc3, '-' )
plt.savefig( "prob.pdf" )
plt.show()

pc1 = []
pc2 = []
with open( "pca.dat", "rt" ) as f:
    f.readline()
    for l in f:
        t = [ float( i ) for i in l.strip().split() ]
        pc1.append( t[1] )
        pc2.append( t[2] )

plt.clf
plt.grid( True )
plt.scatter( pc1, pc2, marker = "o", c = range( len( pc1 ) ) )
plt.xlabel( "PC1" )
plt.ylabel( "PC2" )
plt.title( "Cartesian coordinate PCA [Å]" )
cbar = plt.colorbar()
cbar.set_label( "Time [ps]" )
plt.savefig( "pc1_pc2.pdf" )
plt.show()
