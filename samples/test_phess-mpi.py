import  numpy
import  qm3
import	qm3.engines.mopac
import	qm3.utils._mpi
import	qm3.utils.hessian
import  io

who, cpu = qm3.utils._mpi.init()

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

mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 0, 1 )

sel = numpy.argwhere( mol.actv.ravel() ).ravel()

tsk = [ [] for i in range( cpu ) ]
siz = 0
for i in sel:
    for j in [0, 1, 2]:
        tsk[siz%cpu].append( ( siz, i, j ) )
        siz += 1

dsp = 1.0e-4
hes = numpy.zeros( ( siz, siz ), dtype=numpy.float64 )
for k,i,j in tsk[who]:
    bak = mol.coor[i,j]
    mol.coor[i,j] = bak + dsp
    mol.get_grad()
    gp = mol.grad[sel].ravel()
    mol.coor[i,j] = bak - dsp
    mol.get_grad()
    hes[k,:] = ( gp - mol.grad[sel].ravel() ) / ( 2.0 * dsp )
    mol.coor[i,j] = bak

qm3.utils._mpi.barrier()

if( who == 0 ):
    for i in range( 1, cpu ):
        hes += numpy.array( qm3.utils._mpi.recv_r8( i, siz * siz ) ).reshape( ( siz, siz ) )
    hes = 0.5 * ( hes + hes.T )
    val, vec = qm3.utils.hessian.frequencies( mol, hes )
    print( val[0:7] )
    qm3.utils._mpi.barrier()
    qm3.utils._mpi.stop()
else:
    qm3.utils._mpi.send_r8( 0, hes.ravel().tolist() )
    qm3.utils._mpi.barrier()
    qm3.utils._mpi.stop()

