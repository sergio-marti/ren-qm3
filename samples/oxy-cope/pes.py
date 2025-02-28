#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.minimize
import  pickle

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.xyz_read( open( "reac.xyz" ) )
mol.guess_atomic_numbers()

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

vii = [ mol.indx["A"][1]["C2"], mol.indx["A"][1]["C3"] ]
vjj = [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C6"] ]

f = open( "pes.log", "wt" )

for ii in range( 30 + 1 ):
    for jj in range( 30 + 1 ):
        mol.engines["uii"] = qm3.engines.mmres.distance( 5000, 1.55 + ii * 0.05, vii )
        mol.engines["ujj"] = qm3.engines.mmres.distance( 5000, 3.05 - jj * 0.05, vjj )
        qm3.actions.minimize.fire( mol, gradient_tolerance = 0.5, step_number = 2000 )
        f.write( "%20.10lf%20.10lf%20.10lf\n"%( mol.rval["uii"][1], mol.rval["ujj"][1], mol.rval["qm"] ) )
        f.flush()
        mol.engines.pop( "uii" )
        mol.engines.pop( "ujj" )
        mol.xyz_write( open( "pes.%02d.%02d"%( ii, jj ), "wt" ) )
    mol.xyz_read( open( "pes.%02d.%02d"%( ii, 0 ) ) )

f.close()
