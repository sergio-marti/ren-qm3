import  numpy
import  qm3
import	qm3.engines.delta_ml
import  io
import  sys
import  os
import  pickle
import  matplotlib.pyplot as plt

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
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

net = qm3.engines.delta_ml.network()
with open( cwd + "test_delta-coul.pk", "rb" ) as f:
    net.coef = pickle.load( f )

obj = qm3.engines.delta_ml.coulomb( net, mol, numpy.ones( mol.natm, dtype=numpy.bool_ ), False )

inp = obj.get_info( mol )
tmp = numpy.linalg.norm( inp )
print( inp.shape, round( tmp, 1 ), "/ 5.6" )

tmp = numpy.zeros( ( len( obj.sele ), len( obj.sele ) ) )
c = 0
for i in range( len( obj.sele ) ):
    for j in range( i, len( obj.sele ) ):
        tmp[i,j] = inp[c]
        tmp[j,i] = inp[c]
        c += 1
plt.clf()
plt.imshow( tmp )
plt.tight_layout()
plt.savefig( "delta_coul.pdf" )

net_f, net_g = net.get_grad( inp )
tmp = numpy.linalg.norm( net_g )
print( net_f, net_g.shape, round( tmp, 1 ), "/ 525.8" )

jac = obj.get_jaco( mol )
tmp = numpy.linalg.norm( jac )
print( jac.shape, round( tmp, 1 ), "/ 4.8" )

mol.engines["ml"] = obj
mol.get_grad()
tmp = numpy.linalg.norm( mol.grad )
print( round( tmp, 1 ), "/ 186.7" )

mol.engines["ml"].get_grad = mol.engines["ml"].num_grad
mol.get_grad()
tmp = numpy.linalg.norm( mol.grad )
print( round( tmp, 1 ), "/ 186.7" )


net = qm3.engines.delta_ml.network()
with open( cwd + "test_delta-acsf.pk", "rb" ) as f:
    net.coef = pickle.load( f )

obj = qm3.engines.delta_ml.acsf( net, mol, numpy.ones( mol.natm, dtype=numpy.bool_ ) )
obj.setup()

inp = obj.get_info( mol )
tmp = numpy.linalg.norm( inp )
print( inp.shape, round( tmp, 1 ), "/ 207.0" )

tmp = inp.reshape( ( len( obj.sele ), len( obj.eta2 ) + len( obj.eta5 ) ) )
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow( tmp[:,0:20] )
plt.subplot(1, 2, 2)
plt.imshow( tmp[:,20:] )
plt.tight_layout()
plt.savefig( "delta_acsf.pdf" )

net_f, net_g = net.get_grad( inp )
tmp = numpy.linalg.norm( net_g )
print( net_f, net_g.shape, round( tmp, 1 ), "/ 61.7" )

mol.engines["ml"] = obj

"""
jac = obj.get_jaco( mol )
tmp = numpy.linalg.norm( jac )
print( jac.shape, round( tmp, 1 ), "/" )

mol.get_grad()
tmp = numpy.linalg.norm( mol.grad )
print( round( tmp, 1 ), "/" )
"""

mol.engines["ml"].get_grad = mol.engines["ml"].num_grad
mol.get_grad()
tmp = numpy.linalg.norm( mol.grad )
print( round( tmp, 1 ), "/ 240.9" )
