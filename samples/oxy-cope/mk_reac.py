import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.actions.minimize

mol = qm3.molecule()
mol.pdb_read( open( "pdb" ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
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

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )
print( sqm.sum(), smm.sum() )
mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

qm3.actions.minimize.steepest_descent( mol, gradient_tolerance = 100 )
qm3.actions.minimize.fire( mol )
with open( "reac.pdb", "wt" ) as f:
    mol.pdb_write( f )
