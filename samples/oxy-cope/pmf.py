#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.utils.parallel
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.dynamics
import  pickle
import  zipfile

opar = qm3.utils.parallel.client_fsi( int( sys.argv[1] ) )

mol = qm3.molecule()
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
zzz = zipfile.ZipFile( "neb.zip", "r" )
mol.xyz_read( zzz.open( "node.%02d"%( opar.node ), "r" ), replace = True )
zzz.close()

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

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

ww = [ mol.indx["A"][1]["C2"], mol.indx["A"][1]["C3"], mol.indx["A"][1]["C5"], mol.indx["A"][1]["C6"] ]
kb = 2500
rx = -1.64 + opar.node * 0.0781
mol.engines["umb"] = qm3.engines.mmres.multiple_distance( kb, rx, ww, [ 1.0, -1.0 ] )

mol.fdat = open( "dat.%02d"%( opar.node ), "wt" )
mol.fdat.write( "%12.6lf%12.6lf\n"%( kb, rx ) )
mol.fdat.flush()

# - restart
#if( os.path.isfile( "rst.%02d"%( opar.node ) ) ):
#    with open( "rst.%02d"%( opar.node ), "rb" ) as f:
#        mol.coor = pickle.load( f )
#        mol.velo = pickle.load( f )
#    mol.engines["mm"].update_coor( mol )

def cstep( obj, stp ):
    global  opar
    obj.fdat.write( "%12.6f\n"%( obj.rval["umb"][1] ) )
    obj.fdat.flush()
    if( stp % 1000 == 0 ):
        with open( "rst.%02d"%( opar.node ), "wb" ) as f:
            pickle.dump( obj.coor, f )
            pickle.dump( obj.velo, f )

qm3.actions.dynamics.langevin_verlet( mol, step_size = 0.001, temperature = 300.0,
    gamma_factor = 100.0, print_frequency = 100, step_number = 10000,
    current_step = cstep )

mol.fdat.close()

opar.stop()
