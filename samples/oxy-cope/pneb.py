import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import	qm3.utils.parallel
import  qm3.actions.minimize
import  qm3.actions.neb
import  sys
import  os

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
mol.pdb_read( open( cwd + "reac.pdb" ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( cwd + "oxy-cope.psf" ) )
mol.guess_atomic_numbers()

_psf = openmm.app.charmmpsffile.CharmmPsfFile( cwd + "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( cwd + "oxy-cope.top", cwd + "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )

emm = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "OpenCL" )
eqm = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

mol.engines["mm"] = emm
mol.engines["qm"] = eqm
mol.set_active( sqm )

tmp = qm3.molecule()
tmp.pdb_read( open( cwd + "prod.pdb" ) )

gues = qm3.actions.neb.distribute( 40, [ mol.coor[sqm], tmp.coor[sqm] ] )

ncpu = 5
# mpirun -n 5 python3 pneb.py
opar = qm3.utils.parallel.client_mpi()

ncpu -= 1
chnk = [ [] for i in range( ncpu ) ]
for i in range( len( gues ) ):
    chnk[i%ncpu].append( i )
chnk.insert( 0, [] )
ncpu += 1


# get only QM energy for the NEB
def my_grad():
    mol.rval = []
    mol.func = 0.0
    mol.grad = numpy.zeros( ( mol.natm, 3 ) )
    mol.engines["mm"].get_grad( mol )
    if( "qm" in mol.engines ):
        mol.func = 0.0
        mol.engines["qm"].get_grad( mol )
    mol.grad *= mol.actv.astype( numpy.float64 )
mol.get_grad = my_grad


if( opar.node == 0 ):
    print( chnk )
    sys.stdout.flush()
    obj = qm3.actions.neb.parall( mol, gues, 200, chnk, opar )
    obj.current_step = lambda stp: sys.stdout.flush()
    qm3.actions.minimize.fire( obj, print_frequency = 1, gradient_tolerance = len( gues ) * 0.1 )
    opar.barrier()
    for who in range( 1, ncpu ):
        opar.send_i4( who, [ 0 ] )
else:
    del gues
    flog = open( "borra_log.%d"%( opar.node ), "wt" )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    dime = len( sele )
    nchk = len( chnk[opar.node] )
    size = dime * nchk
    grad = numpy.zeros( ( size, 3 ) )
    opar.barrier()
    flag = opar.recv_i4( 0, 1 )[0]
    while( flag == 1 ):
        # get current coordinates for my chunks
        coor = numpy.array( opar.recv_r8( 0, 3 * size ) )
        coor.shape = ( size, 3 )
        # calculate gradients
        vpot = []
        for who in range( nchk ):
            ii = who * dime
            mol.coor[sele] = coor[ii:ii+dime]
            # ---------------------------------------------------------
            bak = mol.actv
            mol.set_active( numpy.logical_not( mol.actv ) )
            eqm.get_func( mol )
            mol.engines["mm"].update_chrg( mol )
            mol.engines.pop( "qm" )
            qm3.actions.minimize.fire( mol, gradient_tolerance = 2.0, log_file = flog )
            flog.flush()
            mol.chrg[sqm] = 0.0
            mol.engines["mm"].update_chrg( mol )
            mol.engines["qm"] = eqm
            mol.set_active( bak )
            # ---------------------------------------------------------
            mol.get_grad()
            mol.project_gRT()
            vpot.append( mol.func )
            grad[ii:ii+dime] = mol.grad[sele]
            # "neb_data" equivalent
            with open( "node.%02d"%( chnk[opar.node][who] ), "wt" ) as f:
                f.write( "REMARK func = %20.3lf\n"%( mol.func ) )
                mol.pdb_write( f )
        # send my functions and gradients to master
        opar.barrier()
        opar.send_r8( 0, vpot )
        opar.send_r8( 0, grad.ravel().tolist() )
        # wait for more..
        opar.barrier()
        flag = opar.recv_i4( 0, 1 )[0]

opar.stop()
