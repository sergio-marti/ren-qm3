import  os
os.environ["OMP_NUM_THREADS"] = "1"
import	numpy
import	qm3
import  qm3.engines._qmlj
import  io

 
f = io.StringIO( """6

O     1.282255    0.559451    2.778995
H     1.282255    1.316401    2.193113
H     1.282255   -0.197499    2.193113
O     0.349673    0.476493    5.425221
H     1.185602    0.528178    5.886092
H     0.600165    0.505265    4.489875
""" )
mol = qm3.molecule()
mol.xyz_read( f )
mol.guess_atomic_numbers()

mol.chrg = numpy.array( [ -0.8340, 0.4170, 0.4170, -0.8340, 0.4170, 0.4170 ] )

# rmin/2 [Ang]
mol.rmin = numpy.array( [  1.7683, 0.0000, 0.0000,  1.7683, 0.0000, 0.0000 ] )

# sigma [Ang]
#mol.rmin = numpy.array( [ 3.15061, 0.0000, 0.0000, 3.15061, 0.0000, 0.0000 ] )
#mol.rmin *= 0.5612310241546865

# sqrt( epsi ) [sqrt( kJ/mol )]
mol.epsi = numpy.array( [  0.635968, 0.0000, 0.0000, 0.635968, 0.0000, 0.0000 ] )
mol.epsi = numpy.sqrt( mol.epsi )

mol.engines["lj"] = qm3.engines._qmlj.run( mol, [0, 1, 2], [3, 4, 5], [] )

mol.get_grad()
print( round( mol.func, 1 ), "/ 5.1" )
print( mol.grad )
