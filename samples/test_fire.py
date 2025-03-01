import  os
os.environ["OPENMM_CPU_THREADS"] = "1"
import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.mopac
import  qm3.actions.minimize

 
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
smm[sqm] = False
print( sqm.sum(), smm.sum() )
mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 1, 1, sqm, smm )
mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, sel_QM = sqm )

qm3.actions.minimize.fire( mol )

mol.get_grad()
print( round( mol.func, 1 ), "/ -32091.1" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 114.4" )
