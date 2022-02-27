import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines
import  qm3.actions


mol = qm3.molecule()

mol.pdb_read( open( "amber.pdb" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
mol.prmtop_read( open( "amber.prmtop" ) )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "amber.prmtop" )
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
mol.engines.append( qm3.engines.qm3_openmm( _sys, _top, sel_QM = sqm, platform = "OpenCL" ) )
mol.engines.append( qm3.engines.qm3_xtb( mol, 1, 0, sel_QM = sqm, sel_MM = smm ) )

mol.get_grad()
print( mol.func )
print( mol.grad )

qm3.actions.fire( mol )

with open( "last.xyz", "wt" ) as f:
    mol.xyz_write( f )
