#!/usr/bin/env python3
import  os
os.environ["OPENMM_CPU_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
import	sys
import  openmm
import  openmm.app
import  openmm.unit
import  numpy

box  = [ 42.320, 47.736, 43.057 ]

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "start.prmtop" )

_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.5 * openmm.unit.angstrom,
    rigidWater = False,
    implicitSolvent = None,
    switchDistance = 14.0 * openmm.unit.angstrom )

#>> fix prmtop box size (based on VDW radii)
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( box[0], 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, box[1], 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, box[2] ) * openmm.unit.angstrom )


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

#>> NPT [http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat]
#_sys.addForce( openmm.openmm.MonteCarloBarostat( 1.0 * openmm.unit.atmosphere, 300.0, 25 ) )


#>> OpenCL
_sim = openmm.app.Simulation( _top.topology, _sys, _int, openmm.Platform.getPlatformByName( "OpenCL" ) )

#>> CUDA (two cards)
#_sim = openmm.app.Simulation( _top.topology, _sys, _int,
#    openmm.Platform.getPlatformByName( "CUDA" ), { "CudaDeviceIndex": "0,1" } )


#>> load PDB
_sim.context.setPositions( openmm.app.pdbfile.PDBFile( "start.pdb" ).getPositions() )

#>> parse XYZ: OpenMM expects the origin at one of the edges...
#crd = []
#with open( "start.xyz", "rt" ) as f:
#    f.readline(); f.readline()
#    for l in f:
#        t = l.split()
# -------------------------------------------------------------------------------------
#        crd.append( openmm.Vec3( float( t[1] ), float( t[2] ), float( t[3] ) ) )
#_sim.context.setPositions( openmm.unit.quantity.Quantity( crd, openmm.unit.angstrom ) )
# -------------------------------------------------------------------------------------
#        crd.append( [ float( t[1] ), float( t[2] ), float( t[3] ) ] ) 
#crd = numpy.array( crd )
#crd -= numpy.min( crd, axis = 0 )
#crd /= 10.0
#_sim.context.setPositions( crd.tolist() )
# -------------------------------------------------------------------------------------


#>> 100 ps NPT
#_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 100, enforcePeriodicBox = True ) )
#_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 100,
#    time = True, potentialEnergy = True, temperature = True, volume = True ) )
#_sim.step( 100000 )
#print( _sim.context.getState().getPeriodicBoxVectors() )


#>> 10 ns NVT
_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 1000, enforcePeriodicBox = True ) )
_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 1000,
    time = True, potentialEnergy = True, temperature = True ) )
_sim.step( 10000000 )


#>> save coordinates
with open( "last.pdb", "wt" ) as f:
    openmm.app.pdbfile.PDBFile.writeFile( _sim.topology,
        _sim.context.getState( getPositions = True ).getPositions(), f )
