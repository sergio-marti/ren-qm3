import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  os

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
box = numpy.array( [ 25.965, 29.928, 28.080 ] )

who = "CHARMM"
if( len( sys.argv ) > 1 ):
    if( sys.argv[1] in [ "AMBER", "CHARMM", "XML" ] ):
        who = sys.argv[1]
print( ">>", who )


if( who == "AMBER" ):
    # ===================================================================
    # Amber PRMTOP
    mol.pdb_read( open( cwd + "amber.pdb" ) )
    mol.boxl = box
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
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top )
    mol.get_grad()
    print( mol.func )
    assert( numpy.fabs( mol.func - -23032.6527 ) < 0.001 ), "[Amber]: function error"
    print( numpy.linalg.norm( mol.grad ) )
    assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 1158.8760 ) < 0.001 ), "[Amber]: gradient error"

elif( who == "CHARMM" ):
    # ===================================================================
    # Charmm PSF/TOP/PAR
    mol.pdb_read( open( cwd + "charmm.pdb" ) )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( cwd + "charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.charmmparameterset.CharmmParameterSet( cwd + "charmm.top", cwd + "charmm.prm" )
    _sys = _psf.createSystem( _prm,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology )
    mol.get_grad()
    print( mol.func )
    assert( numpy.fabs( mol.func - -23239.8427 ) < 0.001 ), "[Charmm]: function error"
    print( numpy.linalg.norm( mol.grad ) )
    assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 1127.7585 ) < 0.001 ), "[Charmm]: gradient error"

else:
    # ===================================================================
    # Charmm PSF/XML(parmed)
    #import  parmed
    #prm = parmed.charmm.CharmmParameterSet( "charmm.top", "charmm.prm" )
    #print( prm.residues )
    #xml = parmed.openmm.OpenMMParameterSet.from_parameterset( prm )
    #xml.write( "prm.xml" )
    # -------------------------------------------------------------------
    mol.pdb_read( open( cwd + "charmm.pdb" ) )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( cwd + "charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.forcefield.ForceField( cwd + "prm.xml" )
    _sys = _prm.createSystem( _psf.topology,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology )
    mol.get_grad()
    print( mol.func )
    assert( numpy.fabs( mol.func - -23323.1197 ) < 0.001 ), "[xml]: function error"
    print( numpy.linalg.norm( mol.grad ) )
    assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 1127.7546 ) < 0.001 ), "[xml]: gradient error"
