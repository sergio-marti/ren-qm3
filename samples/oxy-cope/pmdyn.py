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
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.engines.metadyn
import  qm3.actions.dynamics
import  qm3.utils._dcd
import  qm3.utils.parallel
import  pickle
import  zipfile

# 6 walkers
opar = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
zzz = zipfile.ZipFile( "s3/otfs.zip", "r" )
mol.xyz_read( zzz.open( f"node.{opar.node * 7:03d}", "r" ), replace = True )
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
mol.engines["qm"].img_init( mol )

with open( "pmf_s.delz", "rt" ) as f:
    dz = float( f.read().strip() )
cvs = qm3.engines.mmres.colvar_s( mol, "pmf_s.cnf", "s3/last.str", delz = dz )
mol.engines["ss"] = qm3.engines.metadyn.run_mw( cvs, opar,
        sigm = 0.2, wght = 2.0, nstp = 100,
        delta = 4000., temperature = 300.,
        name = f"mdyn.wtm" )

mol.engines["w0"] = qm3.engines.mmres.distance( 2500., 3.3, [ 1, 2 ], skip_LE = 3.5 )
mol.engines["w1"] = qm3.engines.mmres.distance( 2500., 3.3, [ 4, 5 ], skip_LE = 3.5 )

mol.dcd = qm3.utils._dcd.dcd()
mol.dcd.open_write( f"mdyn.{opar.node:02d}.dcd", mol.natm )

def cstep( obj, stp ):
    if( stp % 100 == 0 ):
        obj.dcd.append( obj.coor )

qm3.actions.dynamics.seed()
qm3.actions.dynamics.langevin_verlet( mol, step_size = 0.001, temperature = 300.0,
    print_frequency = 100, step_number = 30_000, current_step = cstep )

mol.dcd.close()

if( opar.node == 0 ):
    mol.engines["ss"].save_hills( "mdyn.hills" )
    crd = numpy.linspace( numpy.min( mol.engines["ss"].hill_s ), numpy.max( mol.engines["ss"].hill_s ), 80 )
    pmf = mol.engines["ss"].get_pmf( crd ) / 4.184
    numpy.savetxt( "mdyn.pmf", numpy.column_stack( ( crd.ravel(), pmf.ravel() ) ) )

opar.barrier()
opar.stop()
