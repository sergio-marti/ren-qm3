import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.dynamics
import  qm3.utils._dcd
import  sys
import  os

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

who = int( sys.argv[1].split( "." )[-1] )

mol = qm3.molecule()
mol.pdb_read( open( "node.%02d"%( who ) ) )
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
print( sqm.sum(), smm.sum(), end = " " )
smm = numpy.logical_and( smm, numpy.logical_not( sqm ) )
print( smm.sum() )

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CUDA" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
kmb = 2000.
ref = who * 0.137036
mol.engines["cv"] = qm3.engines.mmres.colvar_s( kmb, ref,
        open( "pmf_s.cnf" ), open( "pmf_s.str" ), open( "pmf_s.met" ) )

dat = open( "dat.%02d"%( who ), "wt" )
dat.write( "%20.10lf%20.10lf\n"%( kmb, ref ) )

dcd = qm3.utils._dcd.dcd()
dcd.open_write( "dcd.%02d"%( who ), mol.natm )

def current_step( stp ):
    dat.write( "%20.10lf\n"%( mol.rval[0] ) )
    if( stp % 100 == 0 ):
        dat.flush()
        dcd.append( mol )
mol.current_step = current_step

qm3.actions.dynamics.langevin_verlet( mol, step_number = 4000 )

dat.close()
dcd.close()
