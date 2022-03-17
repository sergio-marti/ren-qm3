import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.minimize
import  os

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()

mol.pdb_read( open( cwd + "reac.pdb" ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( cwd + "oxy-cope.psf" ) )
mol.guess_atomic_numbers()

_psf = openmm.app.charmmpsffile.CharmmPsfFile( cwd + "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( cwd + "oxy-cope.top", cwd + "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )
print( sqm.sum(), smm.sum() )
mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "OpenCL" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["dst1"] = qm3.engines.mmres.distance( 2500, 3.08, [ mol.indx["A"][1]["C2"], mol.indx["A"][1]["C3"] ] )
mol.engines["dst2"] = qm3.engines.mmres.distance( 2500, 1.55, [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C6"] ] )

qm3.actions.minimize.fire( mol, gradient_tolerance = 10 )
del mol.engines["dst1"]
del mol.engines["dst2"]

qm3.actions.minimize.fire( mol, gradient_tolerance = 3 )
with open( "prod.pdb", "wt" ) as f:
    mol.pdb_write( f )
