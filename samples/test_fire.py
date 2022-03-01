import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
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

sqm = mol.resn == "SUS"
smm = mol.sph_sel( sqm, 10 )
print( sqm.sum(), smm.sum() )
mol.engines.append( qm3.engines.openmm.run( _sys, _top, sel_QM = sqm, platform = "OpenCL" ) )
mol.engines.append( qm3.engines.xtb.run( mol, 1, 0, sel_QM = sqm, sel_MM = smm ) )

qm3.actions.minimize.fire( mol )

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -95302.54253867632 ) < 0.01 ), "Fire[OpenMM/xTB]: function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 114.83926498050597 ) < 0.01 ), "Fire[OpenMM/xTB]: gradient error"
