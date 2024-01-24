#!/usr/bin/env python3
import	sys
import  openmm
import  openmm.app
import  openmm.unit
import  numpy

box  = [ 42.320, 47.736, 43.057 ]

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "start.prmtop" )

_sys = _top.createSystem(
    nonbondedMethod = openmm.app.PME,
    nonbondedCutoff = 14 * openmm.unit.angstrom,
    switchDistance = 12 * openmm.unit.angstrom,
#>> 2 fs
#    constraints = openmm.app.HBonds,
    rigidWater = False,
    implicitSolvent = None )
#    implicitSolvent = openmm.app.HCT, soluteDielectric = 4.0, solventDielectric = 80.0 )


#>> fix prmtop box size (based on VDW radii)
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( box[0], 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, box[1], 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, box[2] ) * openmm.unit.angstrom )


#>> frezee atoms
#for i in range( 55 ):
#    _sys.setParticleMass( i, 0.0 )


#>> add harmonic restraint (remove non-bonding)
#for i in range( _sys.getNumForces() ):
#    cur = _sys.getForce( i )
#    if( type( cur ) == openmm.HarmonicBondForce ):
#        cur.addBond( 35194, 35123,
#            2.0 * openmm.unit.angstrom,
#            400.0 * openmm.unit.kilojoule / ( openmm.unit.angstrom ** 2 * openmm.unit.mole ) )
#    if( type( cur ) == openmm.NonbondedForce ):
#        cur.addException( 35194, 35123, 0.0, 0.0, 0.0, replace = True )


_int = openmm.LangevinIntegrator( 300.0, 5.0, 0.001 )

#>> NPT [http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat]
#_sys.addForce( openmm.openmm.MonteCarloBarostat( 1.0 * openmm.unit.atmosphere, 300.0, 25 ) )


#>> OpenCL
_sim = openmm.app.Simulation( _top.topology, _sys, _int, openmm.Platform.getPlatformByName( "OpenCL" ) )

#>> CUDA (two cards)
#_sim = openmm.app.Simulation( _top.topology, _sys, _int,
#    openmm.Platform.getPlatformByName( "CUDA" ), { "CudaDeviceIndex": "0,1" } )


#>> load PDB (should be centered at the edge...)
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


#_sim.minimizeEnergy()


#>> 100 ps NPT
#_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 100, enforcePeriodicBox = True ) )
#_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 100,
#    time = True, potentialEnergy = True, temperature = True, density = True ) )
#_sim.step( 100000 )
#print( _sim.context.getState().getPeriodicBoxVectors() )


#>> 10 ns NVT
_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 1000, enforcePeriodicBox = True ) )
_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 1000,
    time = True, potentialEnergy = True, temperature = True ) )
_sim.step( 10000000 )

#>> make a context checkpoint for restarting: _sim.context.loadCheckpoint( f.read() )
with open( "last.chk", "wb" ) as f:
    f.write( _sim.context.createCheckpoint() )

#>> save coordinates (OpenMM: PDB)
with open( "last.pdb", "wt" ) as f:
    openmm.app.pdbfile.PDBFile.writeFile( _sim.topology,
        _sim.context.getState( getPositions = True, enforcePeriodicBox = True ).getPositions(), f )

#>> save coordinates (XYZ)
tmp = _sim.context.getState( getPositions = True, enforcePeriodicBox = True ).getPositions()
crd = []
for i in range( len( tmp ) ):
    crd.append( [ tmp[i].x, tmp[i].y, tmp[i].z ] )
crd = numpy.array( crd ) * 10
crd -= numpy.mean( crd, axis = 0 )
box = _sim.context.getState().getPeriodicBoxVectors()
with open( "last.xyz", "wt" ) as f:
    f.write( "%d\n"%( crd.shape[0] ) )
    f.write( "%14.6lf%14.6lf%14.6lf\n"%( box[0].x * 10, box[1].y * 10, box[2].z * 10 ) )
    for i in range( crd.shape[0] ):
        f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( "X", crd[i,0], crd[i,1], crd[i,2] ) )
