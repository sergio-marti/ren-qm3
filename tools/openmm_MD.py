#!/usr/bin/env python3
import	sys
import  openmm
import  openmm.app
import  openmm.unit

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "start.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.5 * openmm.unit.angstrom,
    rigidWater = False,
    implicitSolvent = None,
    switchDistance = 14.0 * openmm.unit.angstrom )
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( 42.320, 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 47.736, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, 43.057 ) * openmm.unit.angstrom )
for i in range( 55 ):
    _sys.setParticleMass( i, 0.0 )
_int = openmm.LangevinIntegrator( 300.0, 50.0, 0.001 )
_sim = openmm.app.Simulation( _top.topology, _sys, _int, openmm.Platform.getPlatformByName( "OpenCL" ) )
_sim.context.setPositions( openmm.app.pdbfile.PDBFile( "start.pdb" ).getPositions() )
#_sim.reporters.append( openmm.app.pdbreporter.PDBReporter( "last.pdb", 1000, enforcePeriodicBox = False ) )
_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 1000, enforcePeriodicBox = False ) )
_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 1000,
    time = True, potentialEnergy = True, temperature = True ) )
_sim.step( 10000 )
