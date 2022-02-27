import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines


mol = qm3.molecule()
box = numpy.array( [ 25.965, 29.928, 28.080 ] )

who = "CHARMM"
if( len( sys.argv ) > 1 ):
    if( sys.argv[1] in [ "AMBER", "CHARMM", "XML" ] ):
        who = sys.argv[1]


if( who == "AMBER" ):
    # ===================================================================
    # Amber PRMTOP
    mol.pdb_read( open( "amber.pdb" ) )
    mol.boxl = box
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
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines.append( qm3.engines.qm3_openmm( _sys, _top ) )

elif( who == "CHARMM" ):
    # ===================================================================
    # Charmm PSF/TOP/PAR
    mol.pdb_read( open( "charmm.pdb" ) )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( "charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.charmmparameterset.CharmmParameterSet( "charmm.top", "charmm.prm" )
    _sys = _psf.createSystem( _prm,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines.append( qm3.engines.qm3_openmm( _sys, _psf.topology ) )

else:
    # ===================================================================
    # Charmm PSF/XML(parmed)
    #import  parmed
    #prm = parmed.charmm.CharmmParameterSet( "charmm.top", "charmm.prm" )
    #print( prm.residues )
    #xml = parmed.openmm.OpenMMParameterSet.from_parameterset( prm )
    #xml.write( "prm.xml" )
    # -------------------------------------------------------------------
    mol.pdb_read( open( "charmm.pdb" ) )
    mol.boxl = box
    _psf = openmm.app.charmmpsffile.CharmmPsfFile( "charmm.psf" )
    _psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
            mol.boxl[1] * openmm.unit.angstrom,
            mol.boxl[2] * openmm.unit.angstrom )
    _prm = openmm.app.forcefield.ForceField( "prm.xml" )
    _sys = _prm.createSystem( _psf.topology,
        nonbondedMethod = openmm.app.CutoffPeriodic,
        nonbondedCutoff = 12.0 * openmm.unit.angstrom,
        switchDistance = 10.0 * openmm.unit.angstrom,
        rigidWater = False )
    print( _sys.getDefaultPeriodicBoxVectors() )
    mol.engines.append( qm3.engines.qm3_openmm( _sys, _psf.topology ) )


mol.get_grad()
print( mol.func )
print( mol.grad )

#import  qm3.actions
#qm3.actions.steepest_descent( mol )
#mol.pdb_write( open( "last.pdb", "wt" ) )
