#!/usr/bin/env python3
import	sys
import  openmm
import  openmm.app
import  openmm.unit
import  numpy

box  = [ 42.320, 47.736, 43.057 ]
# >> parse namd crystal information...
#with open( "namd_npt.xsc", "rt" ) as f:
#    f.readline(); f.readline()
#    t = f.readline().split()
#    box = [ float( t[1] ), float( t[5] ), float( t[9] ) ]
print( box )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "start.prmtop" )

_sys = _top.createSystem(
#    nonbondedMethod = openmm.app.CutoffNonPeriodic, # change "enforcePeriodicBox"
    nonbondedMethod = openmm.app.PME,
    nonbondedCutoff = 14 * openmm.unit.angstrom,
    switchDistance = 12 * openmm.unit.angstrom,
    constraints = None,
#    constraints = openmm.app.HBonds, # 2 fs
#    constraints = openmm.app.AllBonds, # 3 fs
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

#>> add harmonic restraint
#bnd = []
#bnd.append( [ who_i, who_j, dst, kmb ] )
#for I in range( _sys.getNumForces() ):
#    cur = _sys.getForce( I )
#    if( type( cur ) == openmm.HarmonicBondForce ):
#        for i,j,r,k in bnd:
#            cur.addBond( i, j,
#                r * openmm.unit.angstrom,
#                k * openmm.unit.kilojoule / ( openmm.unit.angstrom ** 2 * openmm.unit.mole ) )
#    # remove non-bonding for close (almost bonding) distances...
#    if( type( cur ) == openmm.NonbondedForce ):
#        for i,j,r,k in bnd:
#            cur.addException( i, j, 0.0, 0.0, 0.0, replace = True )

_int = openmm.LangevinIntegrator( 300.0, 5.0, 0.001 )

#>> NPT [http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat]
#_sys.addForce( openmm.openmm.MonteCarloBarostat( 1.0 * openmm.unit.atmosphere, 300.0, 25 ) )

#>> OpenCL
_sim = openmm.app.Simulation( _top.topology, _sys, _int, openmm.Platform.getPlatformByName( "OpenCL" ) )

#>> CUDA (two cards)
#_sim = openmm.app.Simulation( _top.topology, _sys, _int,
#    openmm.Platform.getPlatformByName( "CUDA" ), { "CudaDeviceIndex": "0,1" } )

# -------------------------------------------------------------------------------------
#>> load XML state
#_sim.loadState( "start.xml" )

#>> load PDB (should be centered at the edge...)
#_sim.context.setPositions( openmm.app.pdbfile.PDBFile( "start.pdb" ).getPositions() )

#>> load PDB (not centered)
#crd = openmm.app.pdbfile.PDBFile( "namd_npt.coor" ).getPositions( asNumpy = True )
#crd -= numpy.min( crd, axis = 0 )
#_sim.context.setPositions( crd.tolist() )

#>> parse XYZ: OpenMM expects the origin at one of the edges...
#crd = []
#with open( "start.xyz", "rt" ) as f:
#    f.readline(); f.readline()
#    for l in f:
#        t = l.split()
#        crd.append( [ float( t[1] ), float( t[2] ), float( t[3] ) ] ) 
#crd = numpy.array( crd )
#crd -= numpy.min( crd, axis = 0 )
#crd /= 10.0
#_sim.context.setPositions( crd.tolist() )
# -------------------------------------------------------------------------------------

#_sim.minimizeEnergy( tolerance = 5 * openmm.unit.kilojoule / openmm.unit.mole, maxIterations = 100 )


#>> 1 ns NPT
#_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 100, enforcePeriodicBox = True ) )
#_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 100,
#    time = True, potentialEnergy = True, temperature = True, density = True ) )
#_sim.step( 1000000 )
#print( _sim.context.getState().getPeriodicBoxVectors() )

#>> 100 ns NVT
_sim.context.setStepCount( 0 )
n = 100000000
_sim.reporters.append( openmm.app.dcdreporter.DCDReporter( "last.dcd", 40000, enforcePeriodicBox = True ) )
_sim.reporters.append( openmm.app.statedatareporter.StateDataReporter( sys.stdout, 10000,
    time = True, potentialEnergy = True, temperature = True,
    remainingTime = True, totalSteps = n ) )
_sim.step( n )

_sim.saveState( "last.xml" )

#>> save coordinates (OpenMM: PDB)
#with open( "last.pdb", "wt" ) as f:
#    openmm.app.pdbfile.PDBFile.writeFile( _sim.topology,
#        _sim.context.getState( getPositions = True, enforcePeriodicBox = True ).getPositions(), f )

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
