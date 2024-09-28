import  sys
import	numpy
import	qm3
import  qm3.engines._dynamo
 
mol = qm3.molecule()
mol.xyz_read( open( "dynamo.xyz" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )

mol.engines["qmmm"] = qm3.engines._dynamo.run( "./dynamo.so" )
mol.engines["qmmm"].update_coor( mol )

mol.get_grad()
print( round( mol.func, 1 ), "/ -21895.1" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 1422.8" )
print( mol.grad[0:19,:] )
