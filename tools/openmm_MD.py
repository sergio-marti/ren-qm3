#!/usr/bin/env python3
import  os
os.environ["OPENMM_CPU_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
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


#>> fix prmtop box size (based on VDW radii)
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( 42.320, 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 47.736, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, 43.057 ) * openmm.unit.angstrom )


#>> frezee atoms
#for i in range( 55 ):
#    _sys.setParticleMass( i, 0.0 )


#>> add harmonic restraint
#for i in range( _sys.getNumForces() ):
#    cur = _sys.getForce( i )
#    if( type( cur ) == openmm.HarmonicBondForce ):
#        cur.addBond( 35194, 35123,
#            2.0 * openmm.unit.angstrom,
#            400.0 * openmm.unit.kilojoule / ( openmm.unit.angstrom ** 2 * openmm.unit.mole ) )


_int = openmm.LangevinIntegrator( 300.0, 50.0, 0.001 )


#>> OpenCL
_sim = openmm.app.Simulation( _top.topology, _sys, _int, openmm.Platform.getPlatformByName( "OpenCL" ) )


#>> CUDA (two cards)
#_sim = openmm.app.Simulation( _top.topology, _sys, _int,
#    openmm.Platform.getPlatformByName( "CUDA" ), { "CudaDeviceIndex": "0,1" } )


#>> from PDB
_sim.context.setPositions( openmm.app.pdbfile.PDBFile( "start.pdb" ).getPositions() )


#>> from XYZ
#crd = []
#with open( "start.xyz", "rt" ) as f:
#    f.readline(); f.readline()
#    for l in f:
#        t = l.split()
#        crd.append( openmm.Vec3( float( t[1] ), float( t[2] ), float( t[3] ) ) )
#_sim.context.setPositions( openmm.unit.quantity.Quantity( crd, openmm.unit.angstrom ) )

_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 1000, enforcePeriodicBox = False ) )
_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 1000,
    time = True, potentialEnergy = True, temperature = True ) )

# 10 ns
_sim.step( 10000000 )
