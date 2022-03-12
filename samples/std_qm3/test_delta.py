import  numpy
import  qm3.mol
import	qm3.engines.ml_model
import  io
import  pickle

mol = qm3.mol.molecule()
f = io.StringIO( """17

C       1.643  -0.102  -0.331
C       0.625  -1.049  -0.445
C      -0.797  -0.347   0.580
C      -1.007   0.925   0.033
C       0.027   1.864   0.145
C       1.429   1.160  -0.901
O      -1.761  -1.344   0.438
H      -2.615  -0.944   0.713
H       2.333  -0.190   0.495
H       0.767  -2.024   0.003
H       0.105  -1.087  -1.389
H      -0.328  -0.378   1.567
H      -1.706   1.004  -0.788
H      -0.162   2.835  -0.288
H       0.529   1.923   1.100
H       0.957   1.184  -1.872
H       2.227   1.880  -0.803
""" )
mol.xyz_read( f )
mol.guess_atomic_numbers()

net = qm3.engines.ml_model.np_model()
with open( "../test_delta-coul.pk", "rb" ) as f:
    net.coef = pickle.load( f )

obj = qm3.engines.ml_model.delta_coul( net, mol, list( range( mol.natm ) ), False )
inf = obj.get_info( mol )
tmp = numpy.array( inf )
print( tmp.shape, numpy.linalg.norm( tmp ) )

net_f, net_g = net.get_grad( inf )
net_g = numpy.array( net_g )
print( net_f, net_g.shape, numpy.linalg.norm( net_g ) )

jac = numpy.array( obj.get_jaco( mol ) ).reshape( ( len( net_g ), len( obj.sele ) * 3 ) )
print( jac.shape, numpy.linalg.norm( jac ) )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
obj.get_grad( mol )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
obj.num_grad( mol )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )

net = qm3.engines.ml_model.np_model()
with open( "../test_delta-acsf.pk", "rb" ) as f:
    net.coef = pickle.load( f )

obj = qm3.engines.ml_model.delta_acsf( net, mol, list( range( mol.natm ) ) )
obj.setup( cutx = 8.0, 
    eta2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    eta5 = [0.1, 0.14, 0.19, 0.23, 0.28, 0.32, 0.37, 0.41, 0.46, 0.5] )
inf = obj.get_info( mol )
tmp = numpy.array( inf )
print( tmp.shape, numpy.linalg.norm( tmp ) )

net_f, net_g = net.get_grad( inf )
net_g = numpy.array( net_g )
print( net_f, net_g.shape, numpy.linalg.norm( net_g ) )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
obj.num_grad( mol )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )
