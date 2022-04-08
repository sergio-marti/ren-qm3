import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.actions.minimize
import  os

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()

mol.pdb_read( open( cwd + "amber.pdb" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
mol.prmtop_read( open( cwd + "amber.prmtop" ) )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( cwd + "amber.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 12.0 * openmm.unit.angstrom,
    switchDistance = 10.0 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( mol.boxl[0], 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, mol.boxl[1], 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, mol.boxl[2] ) * openmm.unit.angstrom )

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, platform = "OpenCL" )

qm3.actions.minimize.lbfgsb( mol, print_frequency = 100 )

with open( "last.xyz", "wt" ) as f:
    mol.xyz_write( f )

mol.get_grad()
print( mol.func )
print( numpy.linalg.norm( mol.grad ) )
