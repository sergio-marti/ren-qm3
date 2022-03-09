import  numpy
import  qm3
import	qm3.utils
import  qm3.utils.hessian
import  io
import  sys
import  os
import  pickle

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
f = io.StringIO( """26

C           3.3785117988       -1.5597014566       -2.4704385269
C           3.1122692398       -2.9582320727       -2.8139748305
C           3.8210019909       -3.5780686151       -3.7692285658
C           5.0002519742       -2.9436033696       -4.4419416737
O           4.8704916160       -2.9527729687       -5.8605685753
C           5.2049404639       -1.4943852765       -4.0506684093
C           4.2051596452       -0.8188539011       -3.3327152949
O           6.4035739183       -1.6742746366       -2.6551504734
C           5.8346999982       -2.2686267776       -1.6581345136
C           4.7958304830       -1.6759081138       -0.9210660452
C           2.3706340619       -0.8296317728       -1.6726100872
O           2.3779435694        0.3615329675       -1.3430730790
O           1.3147857008       -1.5752021892       -1.2385244766
C           6.1855655523       -3.6833497553       -1.3475102338
O           5.8761258005       -4.3357269815       -0.3450306891
O           6.9721074595       -4.3093515800       -2.2659538447
H           5.9299913978       -3.5283440088       -4.1529613348
H           4.7255738475       -3.8707805187       -6.1197016355
H           5.9016033543       -0.9315982480       -4.6978623557
H           4.7781845855       -0.5804263514       -0.7947993767
H           4.3007596704       -2.2641328512       -0.1331495493
H           2.2855076360       -3.4534270262       -2.2783875100
H           3.5836246443       -4.6030885517       -4.0909671045
H           4.2069725062        0.2816471597       -3.2777512137
H           0.7132224994       -1.0100977893       -0.7239810064
H           7.2054065864       -5.1951553158       -1.9398695949
""" )
mol.xyz_read( f )
mol.guess_atomic_numbers()

with open( cwd + "test_hess.pk", "rb" ) as f:
    hes = numpy.array( pickle.load( f ) )
    hes.shape = ( 3 * mol.natm, 3 * mol.natm )
    mol.chrg = numpy.array( pickle.load( f ) )
    dsp = numpy.array( pickle.load( f ) )
    grd = numpy.array( pickle.load( f ) )

tmp = numpy.trace( hes )
print( tmp )
assert( numpy.fabs( tmp - 273925.989 ) < 0.01 ), "Hessian calculation error"

tmp = numpy.linalg.norm( numpy.sum( mol.mass * mol.coor * mol.actv, axis = 0 ) / numpy.sum( mol.mass * mol.actv ) )
print( tmp )
assert( numpy.fabs( tmp - 5.6317 ) < 0.001 ), "Center of mass error"

rtm = qm3.utils.RT_modes( mol )
tmp = rtm[-1].sum()
print( tmp )
assert( numpy.fabs( tmp - -0.0191 ) < 0.001 ), "RT modes error"

val, vec = qm3.utils.hessian.frequencies( mol, qm3.utils.hessian.raise_RT( hes, rtm ) )
print( val[0:7] )
tmp = numpy.linalg.norm( val )
print( tmp )
assert( numpy.fabs( tmp - 13993.7140 ) < 0.001 ), "raise RT modes error"

val, vec = qm3.utils.hessian.frequencies( mol, hes )
print( val[0:7] )
tmp = numpy.linalg.norm( val )
print( tmp )
assert( numpy.fabs( tmp - 13112.6629 ) < 0.001 ), "Frequencies error"
tmp = numpy.linalg.norm( vec[:,-1] )
print( tmp )
assert( numpy.fabs( tmp - 0.9961 ) < 0.001 ), "Normal Modes error"

iri = qm3.utils.hessian.IR_intensities( mol, vec )
tmp = numpy.linalg.norm( iri[6:] )
print( tmp )
assert( numpy.fabs( tmp - 463.80 ) < 0.1 ), "IR intensities error"

qm3.utils.hessian.IR_spectrum( val, iri )

rms, frc = qm3.utils.hessian.force_constants( mol, val, vec )
tmp = numpy.linalg.norm( rms[6:] )
print( tmp )
assert( numpy.fabs( tmp - 39.4503 ) < 0.001 ), "Reduced masses error"

qm3.utils.hessian.normal_mode( mol, val, vec, 0, afac = 8.0 )

zpe, gib = qm3.utils.hessian.rrho( mol, val )
print( zpe )
assert( numpy.fabs( zpe - 518.0466 ) < 0.001 ), "ZPE error"
print( gib )
assert( numpy.fabs( gib - -110.7683 ) < 0.001 ), "Gibbs error"

bak = hes.copy()
for func, check in [ ( qm3.utils.hessian.update_bfgs, 13120.3661 ),
        ( qm3.utils.hessian.update_psb, 13112.7332 ),
        ( qm3.utils.hessian.update_bofill, 13112.7567 ) ]:
    hes = bak.copy()
    func( dsp, grd, hes )
    val, vec = qm3.utils.hessian.frequencies( mol, hes )
    tmp = numpy.linalg.norm( val )
    print( val[0], tmp )
    assert( numpy.fabs( tmp - check ) < 0.001 ), "Hessian update error"

mol.grad = grd.reshape( ( mol.natm, 3 ) )
qm3.utils.hessian.manage( mol, bak )
