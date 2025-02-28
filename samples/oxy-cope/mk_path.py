#!/usr/bin/env python3
import  numpy
import  qm3.utils.grids
import  matplotlib.pyplot as plt

g = qm3.utils.grids.grid()
g.parse( open( "pes.reg" ) )

levels = 40

plt.clf()
plt.grid()
ly, lx = numpy.meshgrid( g.y, g.x )
lz = g.z.reshape( ( g.x.shape[0], g.y.shape[0] ) )
z_min = min( g.z )
z_max = max( g.z )
z_lvl = [ z_min + ( z_max - z_min ) / float( levels ) * float( i ) for i in range( levels + 1 ) ]
plt.contourf( lx, ly, lz, levels = z_lvl, cmap = "coolwarm" )
cntr = plt.contour( lx, ly, lz, levels = z_lvl, colors = "black", linewidths = 2 )
plt.clabel( cntr, inline = True, levels = z_lvl, fontsize = 7, fmt = "%.1lf" )

c = numpy.loadtxt( "path.log" )
plt.plot( c[:,0], c[:,1], '-og' )

plt.savefig( "path.pdf" )

p = []
for i in range( c.shape[0] ):
    t = "pes.%02d.%02d"%( int( round( ( c[i,0] - 1.55 ) / 0.05, 0 ) ), int( round( ( 3.05 - c[i,1] ) / 0.05, 0 ) ) )
    if( not t in p ):
        p.append( t )
with open( "path.nodes", "wt" ) as f:
    f.write( "\n".join( p ) )
