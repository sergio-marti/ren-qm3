#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.engines.hyperplane
import  qm3.actions.dynamics
import  qm3.actions.hts_dynamics
import  qm3.utils.parallel
import  qm3.utils._dcd
import  qm3.utils.hessian
import  pickle
import  sys

"""
Hiperplano de transicion sobre las distancias de una colvar_s ( plantilla: adaptar build_system() )

  python hts_workflow.py optimize     # 1 proceso por walker ( client_fsi ): UNICA etapa con sesgo cuadratico
  python hts_workflow.py produce      # dinamica en q = 0 SIN ningun restraint ( constraint RATTLE )
"""

STAGE, NWALK = ( sys.argv[1] if len( sys.argv ) > 1 else "optimize" ), 4

opar = qm3.utils.parallel.client_fsi( NWALK ) if STAGE == "optimize" else None
node = opar.node if opar is not None else 0
qm3.actions.dynamics.seed()

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
mol.xyz_read( open( "reac.xyz", "rt" ), replace = True )
reac = mol.copy()
mol.xyz_read( open( "saddle.xyz", "rt" ), replace = True )

_psf = openmm.app.charmmpsffile.CharmmPsfFile( "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( "oxy-cope.top", "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["qm"].img_init( mol )

hts = qm3.engines.hyperplane.colvar_hts( mol, "pmf_s.cnf" )

if( STAGE == "optimize" ):
    mol.set_active( sqm )
    hess = qm3.utils.hessian.numerical( mol )
    mol.set_active()
    info = hts.initial_guess_ts( mol, hess, sele = sqm, react = reac )
    print( info ) # overlap ~ 1: las distancias expresan el modo imaginario; si es bajo, faltan CVs
    kmb = 3000.0 # ps^-2
    bia = qm3.engines.hyperplane.bias( hts, kmb, dmax = 1.0 )
    mol.engines["hts"] = bia
    opt = qm3.engines.hyperplane.optimizer( hts, kmb, temperature = 300., opar = opar, log_file = open( "hts_conv.log", "wt" ) if node == 0 else None )
    dt, nsync, ttot, tconv = 0.0005, 10, 8.0, 5.0
    stp = qm3.actions.dynamics.stepper__langevin_verlet( mol, dt, 300., 50.0 )
    dcd = qm3.utils._dcd.dcd()
    dcd.open_write( "hts_conv.%03d.dcd"%( node ), mol.natm )
    for it in range( int( round( ttot / ( dt * nsync ) ) ) ):
        for k in range( nsync ):
            stp.integrate( mol )
            opt.collect( bia.sample )
        opt.update( dt * nsync )
        if( opt.time >= tconv and not opt.store ):
            opt.start_storing( 10 )
        dcd.append( mol.coor )
    opt.final( tconv )
    dcd.close()
    if( node == 0 ):
        hts.save( "hts.json" )
        print( "importancia de cada distancia ( ~1 irrelevante ):", opt.importance()[0] )
    opar.barrier()
    opar.stop()
else:
    mol.dcd = qm3.utils._dcd.dcd()
    mol.dcd.open_write( "hts_md.dcd", mol.natm )
    hts.load( "hts.json" )
    def keep( m, i ):
        if( i % 100 == 0 ):
            m.dcd.append( m.coor )
    qm3.actions.hts_dynamics.constrained_verlet( mol, hts, step_size = 0.0005, temperature = 300., temperature_coupling = 0.1,
        step_number = 20_000, print_frequency = 100, current_step = keep )
    mol.dcd.close()
