import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.xtb
import  qm3.engines._qmlj
import  qm3.utils.hessian
import  pickle
import  sys

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.xyz_read( open( sys.argv[1] ) )
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

mol.epsi = numpy.zeros( mol.natm )
mol.rmin = numpy.zeros( mol.natm )
for k in range( _sys.getNumForces() ):
    c = _sys.getForce( k )
    if( type( c ) == openmm.NonbondedForce ):
        for i in range( mol.natm ):
            q,s,e = c.getParticleParameters( i )
            mol.rmin[i] = s.value_in_unit( openmm.unit.angstrom )
            mol.epsi[i] = e.value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
mol.epsi = numpy.sqrt( mol.epsi )
mol.rmin *= 0.5612310241546865

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["lj"] = qm3.engines._qmlj.run( mol,
                        numpy.flatnonzero( sqm ).tolist(),
                        numpy.flatnonzero( smm ).tolist(),
                        [] )

mol.set_active( sqm )

hes = qm3.utils.hessian.par_numerical( 4, mol, central = True )

val, vec = qm3.utils.hessian.frequencies( mol, hes )
print( val[0:7] )
qm3.utils.hessian.normal_mode( mol, val, vec, 0 )
