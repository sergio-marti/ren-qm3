#!/usr/bin/env python3
import  sys
import  numpy
import  scipy
import  skimage
import  pyvista
import  qm3
import  qm3.utils._conn


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
        for i in range( n ):
            f.readline()
        o = numpy.array( [ float( t ) for t in f.read().split() ] )
        return( l0, ( nx, ny, nz ), ( dx, dy, dz ), o )


who = sys.argv[1]

v, n, d, den = parse_cube( who + ".den.cube" )
den.shape = n
vert, face, norm, valu = skimage.measure.marching_cubes( den, level = 0.04 )
verT = numpy.empty_like( vert )
verT[:, 0] = v[0] + vert[:, 0] * d[0]
verT[:, 1] = v[1] + vert[:, 1] * d[1]
verT[:, 2] = v[2] + vert[:, 2] * d[2]

v, n, d, esp = parse_cube( who + ".esp.cube" )
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

facx = numpy.hstack( ( numpy.full( ( face.shape[0], 1 ), 3 ), face ) ).astype( numpy.int32 )
mesh = pyvista.PolyData( verT, facx )
mesh.point_data["MEP"] = col

mol = qm3.molecule()
mol.mol2_read( open( sys.argv[1] + ".mol2" ) )
mol.labl = numpy.array( [ a.title() for a in mol.labl ] )
mol.guess_atomic_numbers()

colors = { 1: "white",
           5: "darkseagreen", 6: "gray", 7: "blue", 8: "red", 9: "lightgreen",
          15: "orange", 16: "yellow", 17: "green",
          35: "darkred", 53: "purple" }

bonds = qm3.utils._conn.connectivity( 2, mol.anum, mol.coor )
v_atm = pyvista.MultiBlock()
c_atm = []
for i in numpy.flatnonzero( mol.actv ):
    v_atm.append( pyvista.Sphere( radius=qm3.data.r_vdw[mol.anum[i]]*0.1, center=mol.coor[i] ) )
    c_atm.append( colors.get( mol.anum[i], "magenta" ) )
v_bnd = pyvista.MultiBlock()
c_bnd = []
for i,j in bonds:
    if( mol.actv[i] and mol.actv[j] ):
        p1, p2 = mol.coor[i], mol.coor[j]
        mid = ( p1 + p2 ) / 2
        vec = mid - p1
        siz = numpy.linalg.norm( vec )
        v_bnd.append( pyvista.Cylinder( center=p1+vec/2, direction=vec, height=siz, radius=0.1 ) )
        c_bnd.append( colors.get( mol.anum[i], "magenta" ) )
        vec = p2 - mid
        siz = numpy.linalg.norm( vec )
        v_bnd.append( pyvista.Cylinder( center=mid+vec/2, direction=vec, height=siz, radius=0.1 ) )
        c_bnd.append( colors.get( mol.anum[j], "magenta" ) )

plot = pyvista.Plotter()
for i in range( len( v_atm ) ):
    plot.add_mesh( v_atm[i], color=c_atm[i], smooth_shading=True )
for i in range( len( v_bnd ) ):
    plot.add_mesh( v_bnd[i], color=c_bnd[i], smooth_shading=True )

plot.add_mesh( mesh, scalars="MEP", cmap="bwr_r", clim=( col_min, col_max ), show_edges=False, smooth_shading=True, opacity=0.8 )

plot.show()
