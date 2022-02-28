import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3.mol
import  qm3.engines.openmm


mol = qm3.mol.molecule()
box = [ 25.965, 29.928, 28.080 ]

who = "CHARMM"
if( len( sys.argv ) > 1 ):
    if( sys.argv[1] in [ "AMBER", "CHARMM", "XML" ] ):
        who = sys.argv[1]
print( ">>", who )


if( who == "AMBER" ):
    # ===================================================================
    # Amber PRMTOP
    mol.pdb_read( "../amber.pdb" )
    mol.boxl = box
    _top = openmm.app.amberprmtopfile.AmberPrmtopFile( "../amber.prmtop" )
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
    eng = qm3.engines.openmm.run_native( _sys, _top )
    mol.func = 0
    mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
    eng.get_grad( mol )
    print( mol.func )
    print( numpy.linalg.norm( mol.grad ) )

elif( who == "CHARMM" ):
    # ===================================================================
    # Charmm PSF/TOP/PAR
    mol.pdb_read( "../charmm.pdb" )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( "../charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.charmmparameterset.CharmmParameterSet( "../charmm.top", "../charmm.prm" )
    _sys = _psf.createSystem( _prm,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    eng = qm3.engines.openmm.run_native( _sys, _psf.topology )
    mol.func = 0
    mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
    eng.get_grad( mol )
    print( mol.func )
    print( numpy.linalg.norm( mol.grad ) )

else:
    # ===================================================================
    # Charmm PSF/XML(parmed)
    #import  parmed
    #prm = parmed.charmm.CharmmParameterSet( "charmm.top", "charmm.prm" )
    #print( prm.residues )
    #xml = parmed.openmm.OpenMMParameterSet.from_parameterset( prm )
    #xml.write( "prm.xml" )
    # -------------------------------------------------------------------
    mol.pdb_read( "../charmm.pdb" )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( "../charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.forcefield.ForceField( "../prm.xml" )
    _sys = _prm.createSystem( _psf.topology,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    eng = qm3.engines.openmm.run_native( _sys, _psf.topology )
    mol.func = 0
    mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
    eng.get_grad( mol )
    print( mol.func )
    print( numpy.linalg.norm( mol.grad ) )
