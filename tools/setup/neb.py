#!/usr/bin/env python3
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.mopac
import  qm3.engines.mmres
import  qm3.actions.minimize
import  qm3.actions.neb
import  qm3.utils.parallel
import  os
import  pickle
import  sys


par = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop" ) )
mol.xyz_read( open( "mini_i.xyz", "rt" ), replace = True )
with open( "namd_npt.xsc" ) as f:
    l = f.readline()
    while( l[0] == "#" ):
        l = f.readline()
    tmp = [ float( s ) for s in l.split() ]
    mol.boxl = numpy.array( [ tmp[1], tmp[5], tmp[9] ] )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "complex.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )
_sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3( mol.boxl[0], 0.0, 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, mol.boxl[1], 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, 0.0, mol.boxl[2] ) * openmm.unit.angstrom )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

emm = qm3.engines.openmm.run( _sys, _top, sqm, "CPU" )
mol.engines["mm"] = emm

with open( "sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

eqm = qm3.engines.mopac.run( mol, "AM1", 0, 1, sqm, smm, sla )
mol.engines["qm"] = eqm

mol.set_active( sqm )

tmp = qm3.molecule()
tmp.xyz_read( open( "mini_r.xyz", "rt") )
gue = [ tmp.coor[sqm] ]

gue.append( mol.coor[sqm] )

tmp.xyz_read( open( "mini_p.xyz", "rt") )
gue.append( tmp.coor[sqm] )

gue = qm3.actions.neb.distribute( 84, gue )
if( par.node == 0 ):
    print( len( gue ) )
    sys.stdout.flush()

cpu = par.ncpu - 1
chk = [ [] for i in range( cpu ) ]
for i in range( len( gue ) ):
    chk[i%cpu].append( i )
chk.insert( 0, [] )

if( par.node == 0 ):
    print( chk )
#sys.exit(1)

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


if( par.node == 0 ):
    print( chk )
    sys.stdout.flush()
    obj = qm3.actions.neb.parall( mol, gue, 200, chk, par )
    obj.current_step = lambda i: sys.stdout.flush()
    qm3.actions.minimize.fire( obj, print_frequency = 1, gradient_tolerance = 12 )#len( gue ) * 0.1 )
    par.barrier()
    for who in range( 1, par.ncpu ):
        par.send_i4( who, [ 0 ] )
else:
    del gue
    flog = open( "borra.%03d.log"%( par.node ), "wt" )
    sele = numpy.argwhere( sqm.ravel() ).ravel()
    dime = len( sele )
    nchk = len( chk[par.node] )
    size = dime * nchk
    grad = numpy.zeros( ( size, 3 ) )
    par.barrier()
    flag = par.recv_i4( 0, 1 )[0]
    while( flag == 1 ):
        # get current coordinates for my chunks
        coor = numpy.array( par.recv_r8( 0, 3 * size ) ).reshape( ( size, 3 ) )
        # calculate gradients
        vpot = []
        for who in range( nchk ):
            ii = who * dime
            mol.coor[sele] = coor[ii:ii+dime]
            # ---------------------------------------------------------
            mol.set_active( smm )
            eqm.get_func( mol )
            mol.engines["mm"].update_chrg( mol )
            mol.engines.pop( "qm" )
            qm3.actions.minimize.fire( mol, gradient_tolerance = 2.0, log_file = flog )
            flog.flush()
            mol.chrg[sqm] = 0.0
            mol.engines["mm"].update_chrg( mol )
            mol.engines["qm"] = eqm
            mol.set_active( sqm )
            # ---------------------------------------------------------
            mol.get_grad()
            mol.project_gRT()
            vpot.append( mol.func )
            grad[ii:ii+dime] = mol.grad[sele]
            # "neb_data" equivalent
            with open( "node.%03d"%( chk[par.node][who] ), "wt" ) as f:
                mol.xyz_write( f, comm = "func = %20.3lf\n"%( mol.func ) )
        # send my functions and gradients to master
        par.barrier()
        par.send_r8( 0, vpot )
        par.send_r8( 0, grad.ravel().tolist() )
        par.barrier()
        flag = par.recv_i4( 0, 1 )[0]

par.stop()
