#!/usr/bin/env python3
import  sys
import  numpy
import  scipy
import  skimage
import  pyvista
import  io
import  qm3
import  qm3.data


#
#for ff in *.fchk; do
#	gg=`basename $ff .fchk`
#	cubegen 0 Density=SCF   $ff $gg.den.cube 0 h
#	cubegen 0 Potential=SCF $ff $gg.esp.cube 0 h
#done
#


def parse_cube( fname ):
    a0 = 0.52917721093
    with open( fname, "rt" ) as f:
        f.readline(); f.readline()
        t = f.readline().split(); n  = int( t[0] ); l0 = numpy.array( [ float( t[1] ), float( t[2] ), float( t[3] ) ] ) * a0
        t = f.readline().split(); nx = int( t[0] ); dx = float( t[1] ) * a0
        t = f.readline().split(); ny = int( t[0] ); dy = float( t[2] ) * a0
        t = f.readline().split(); nz = int( t[0] ); dz = float( t[3] ) * a0
        b = "%d\n\n"%( n )
        for i in range( n ):
            t = f.readline().split()
            b += "%s%12.6lf%12.6lf%12.6lf\n"%( qm3.data.symbol[int(t[0])], float( t[2] ) * a0, float( t[3] ) * a0, float( t[4] ) * a0 )
        o = numpy.array( [ float( t ) for t in f.read().split() ] )
        return( b, l0, ( nx, ny, nz ), ( dx, dy, dz ), o )


who = sys.argv[1]

b, v, n, d, den = parse_cube( who + ".den.cube" )
den.shape = n
vert, face, norm, valu = skimage.measure.marching_cubes( den, level = 0.04 )
verT = numpy.empty_like( vert )
verT[:, 0] = v[0] + vert[:, 0] * d[0]
verT[:, 1] = v[1] + vert[:, 1] * d[1]
verT[:, 2] = v[2] + vert[:, 2] * d[2]

esp = parse_cube( who + ".esp.cube" )[-1]
esp.shape = n

col = scipy.ndimage.map_coordinates( esp, vert.T, order=1, mode='nearest' )
xol = numpy.mean( col[face], axis = 1 )
x, y = numpy.histogram( xol, bins = int( numpy.round( 100 * ( numpy.max( xol ) - numpy.min( xol ) ), 0 ) ) )
w = numpy.argmax( x )
i = w
while( i < x.shape[0] and x[i] > 0.01 * x[w]  ):
    i += 1
j = w
while( j > 0 and x[j] > 0.01 * x[w] ):
    j -= 1
avr = numpy.mean( y[j:i+1] )
rms = numpy.std( y[j:i+1] )
col_min = max( numpy.min( xol ), avr - 2 * rms )
col_max = min( numpy.max( xol ), avr + 2 * rms )
print( col_min, col_max )

#col_min = 0.13866256182161965
#col_max = 0.48106167595525307

facx = numpy.hstack( ( numpy.full( ( face.shape[0], 1 ), 3 ), face ) ).astype( numpy.int32 )
mesh = pyvista.PolyData( verT, facx )
mesh.point_data["MEP"] = col

mol = qm3.molecule()
mol.xyz_read( io.StringIO( b ) )
mol.guess_atomic_numbers()

plot = qm3.vBS( mol, display = False )
plot.add_mesh( mesh, scalars="MEP", cmap="bwr_r", clim=( col_min, col_max ), show_edges=False, smooth_shading=True, opacity=0.9 )
plot.show()
