#!/usr/bin/env python3
import  sys
import  numpy
import  torch
import  qm3
import  qm3.data
import  qm3.utils
import  qm3.utils._dcd
import  qm3.engines.mopac
import  qm3.actions.dynamics
import  qm3.engines.mlmodel
import  io
import  os

mol = qm3.molecule()
f = io.StringIO( """9

C      -0.00107      0.00464      0.00498
C       0.88287      0.85412      0.89805
H      -0.64469     -0.64806      0.60245
H      -0.63171      0.63272     -0.63057
H       0.60559     -0.64487     -0.63312
H       0.28222      1.49202      1.55333
H       1.53075      1.49589      0.29439
O       1.71071      0.02145      1.70302
H       1.12820     -0.52903      2.25450
""" )
mol.xyz_read( f )
mol.guess_atomic_numbers()
mol.xyz_write( open( "xyz", "wt" ) )
mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 0, 1 )
mol.get_grad()
o_func = mol.func
o_grad = mol.grad.copy()

dev = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( dev )
ref = mol.coor - numpy.mean( mol.coor, axis = 0 )

# ------------------------------------------------------------------------
# build dataset
#
if( "build" in sys.argv or not os.path.isfile( "dcd" ) ):
    ene = []
    crd = []
    grd = []
    mol.dcd = qm3.utils._dcd.dcd()
    mol.dcd.open_write( "dcd", mol.natm )
    def cstep( obj, stp ):
        if( stp % 10 == 0 ):
            ene.append( obj.func )
            crd.append( obj.coor.copy() )
            grd.append( obj.grad.copy() )
            obj.dcd.append( obj )
    qm3.actions.dynamics.langevin_verlet( mol, step_number = 40000, print_frequency = 100, current_step = cstep )
    mol.dcd.close()
    ene = numpy.array( ene )
    crd = numpy.array( crd )
    grd = numpy.array( grd )

mol.engines.pop( "qm" )

# ------------------------------------------------------------------------
# preprocess dataset
#
if( "process" in sys.argv or not os.path.isfile( "eref" ) ):
    #rng = numpy.array( [ numpy.min( ene ) - 4.184 * 3, numpy.max( ene ) + 4.184 * 3 ] )
    rng = numpy.array( [ numpy.mean( ene ), numpy.std( ene ) ] )
    numpy.savetxt( "eref", rng )
    print( rng )
    ene = torch.tensor( ene, dtype=torch.float32 ).to( dev ).unsqueeze( -1 )
    torch.save( ene, "ene.pt" )
    #: env = qm3.engines.mlmodel.ebuild( crd, torch.zeros( crd.shape[1], dtype=torch.float32 ).to( dev ), 8 ).to( dev )
    #: torch.save( env, "env.pt" )
    #: print( env )
    #: net = [ env.shape[1] * env.shape[1], 256, 256, 128 ]
    #: print( net )
    #: with open( "network", "wt" ) as f:
    #:     f.write( " ".join( [ str( i ) for i in net ] ) + "\n" )
    crd = torch.tensor( crd, dtype=torch.float32 ).to( dev )
    torch.save( crd, "crd.pt" )
    grd = torch.tensor( grd, dtype=torch.float32 ).to( dev )
    torch.save( grd, "grd.pt" )
else:
    rng = numpy.loadtxt( "eref" )
    #: with open( "network", "rt" ) as f:
    #:     net = [ int( i ) for i in f.readline().split() ]
    #: env = torch.load( "env.pt", map_location=dev, weights_only=True )
    ene = torch.load( "ene.pt", map_location=dev, weights_only=True )
    crd = torch.load( "crd.pt", map_location=dev, weights_only=True )
    grd = torch.load( "grd.pt", map_location=dev, weights_only=True )

#ene = ( ene - rng[0] ) / ( rng[1] - rng[0] ) * 2.0 - 1.0
#grd = grd * 2.0 / ( rng[1] - rng[0] )
ene = ( ene - rng[0] ) / rng[1]
grd /= rng[1]
#: print( env.shape, ene.shape, crd.shape, grd.shape )
print( ene.shape, crd.shape, grd.shape )

# ------------------------------------------------------------------------
# train
#
#: mol.engines["ml"] = qm3.engines.mlmodel.run( rng, env,
#:                         numpy.ones( mol.natm, dtype=numpy.bool_ ),
#:                         [ qm3.data.symbol[i] for i in mol.anum ], net, dev )
mol.engines["ml"] = qm3.engines.mlmodel.run_egnn( rng, numpy.ones( mol.natm, dtype=numpy.bool_ ), mol.anum, dev, [100, 32, 64, 3, 5.0] )

#: if( "train" in sys.argv or not os.path.isfile( "atom_C.pth" ) ):
if( "train" in sys.argv or not os.path.isfile( "_egnn_model.pth" ) ):
    #bsize = ene.shape[0]
    bsize = 128
    nepoc = 99999
    optim = torch.optim.Adam( mol.engines["ml"].parameters(), lr = 2.e-4 )
    ##sched = qm3.engines.mlmodel.scheduler( optim, min_lr = 1.e-7, max_lr = 1.e-3, steps_per_epoch = 1, lr_decay = 0.9, cycle_length = 100, mult_factor = 1.5 )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau( optim, mode = 'min', factor = 0.5, patience = 15, min_lr = 1.e-7 )
    lossf = torch.nn.MSELoss()
    dset  = torch.utils.data.TensorDataset( crd, ene, grd )
    vsiz  = int( 0.2 * len( dset ) )
    tsiz  = len( dset ) - vsiz
    tset, vset = torch.utils.data.random_split( dset, [ tsiz, vsiz ], generator = torch.Generator().manual_seed( 69 ) )
    tload = torch.utils.data.DataLoader( tset, batch_size = bsize, shuffle = True )
    vload = torch.utils.data.DataLoader( vset, batch_size = bsize, shuffle = False )

    #scal  = ( rng[1] - rng[0] ) / ( 2.0 * 4.184 ) # kcal/mol(Ang)
    scal  = rng[1] / 4.184
    blos  = float( "inf" )
    for epoch in range( nepoc ):
        tlos = 0.0
        for bx, bf, bg in tload:
            optim.zero_grad()
            bx.requires_grad_( True )
            #: bi = qm3.engines.mlmodel.xcoul_info( bx, env )
            #: func = mol.engines["ml"]( bi )
            anum = torch.tensor(mol.anum, dtype=torch.long, device=dev).unsqueeze(0).expand(bx.shape[0], -1)
            func = mol.engines["ml"]( anum, bx )
            grad = torch.autograd.grad( func.sum(), bx, create_graph=True )[0]
            flos = lossf( func, bf )
            glos = lossf( grad, bg )
            loss = flos + 10 * glos
            loss.backward()
            optim.step()
            tlos += loss.item()
        tlos = tlos / len( tload )
        vene = 0.0
        vgrd = 0.0
        for bx, bf, bg in vload:
            bx.requires_grad_( True )
            #: bi = qm3.engines.mlmodel.xcoul_info( bx, env )
            #: func = mol.engines["ml"]( bi )
            anum = torch.tensor(mol.anum, dtype=torch.long, device=dev).unsqueeze(0).expand(bx.shape[0], -1)
            func = mol.engines["ml"]( anum, bx )
            grad = torch.autograd.grad( func.sum(), bx, create_graph=False )[0]
            flos = lossf( func, bf )
            glos = lossf( grad, bg )
            vene += flos.item()
            vgrd += glos.item()
        vene = vene / len( vload )
        vgrd = vgrd / len( vload )
        vlos = vene + 10.0 * vgrd
        sched.step( vlos )
        rene = numpy.sqrt( vene ) * scal
        rgrd = numpy.sqrt( vgrd ) * scal
        print( f"Epoch {epoch+1:5d}/{nepoc:5d}, RMS_ene: {rene:8.4f}, RMS_grd: {rgrd:8.4f}, VLoss: {vlos:8.4f}", end = "" )
        if( vlos < blos ):
            mol.engines["ml"].save()
            print( " << Checkpoint" )
            blos = vlos
        else:
            print()
        if( rgrd <= 1.5 ):
            break
        sys.stdout.flush()

# ------------------------------------------------------------------------
# check
#
mol.engines["ml"].load()
mol.get_grad()
print( "[AM1] -------------------------------------------------------" )
print( o_func )
print( o_grad )
#: print( "[ML] --------------------------------------------------------" )
print( "[ML (EGNN)] -------------------------------------------------" )
print( mol.func )
print( mol.grad )
print( "-------------------------------------------------------------" )
print( "func: %.1lf%%"%( ( mol.func - o_func ) / o_func * 100 ) )
print( "grad:", numpy.linalg.norm( o_grad - mol.grad ) )
