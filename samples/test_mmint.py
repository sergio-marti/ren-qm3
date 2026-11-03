import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.utils._conn
import  qm3.engines
import  qm3.engines._mmint
import  qm3.engines._qmlj
import  os
import  sys

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( cwd + "amber.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffNonPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom )

mol = qm3.molecule()
mol.prmtop_read( open( cwd + "amber.prmtop" ) )
mol.pdb_read( open( cwd + "amber.pdb" ) )
print( mol.anum )
print( mol.chrg )

for k in range( _sys.getNumForces() ):
    cur = _sys.getForce( k )
    if( type( cur ) == openmm.openmm.NonbondedForce ):
        mol.epsi = numpy.zeros( mol.natm, dtype=numpy.float64 )
        mol.rmin = numpy.zeros( mol.natm, dtype=numpy.float64 )
        for i in range( mol.natm ):
            q,s,e = cur.getParticleParameters( i )
            mol.rmin[i] = s.value_in_unit( openmm.unit.angstrom )
            mol.epsi[i] = e.value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
        mol.rmin *= 0.5612310241546865
        mol.epsi = numpy.sqrt( mol.epsi )
print( mol.epsi )
print( mol.rmin )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
smm[sqm] = False
print( sqm.sum(), smm.sum() )

exc = qm3.engines.exclusions( mol.natm, qm3.utils._conn.connectivity( 4, mol.anum, mol.coor ), sqm )
print( exc )
l_qm = numpy.flatnonzero( sqm ).tolist()
l_mm = numpy.flatnonzero( smm ).tolist()
mol.engines["mm"] = qm3.engines._mmint.run( mol, l_qm, l_mm, exc )

mol.get_grad()
print( round( mol.func, 1 ), "/ -26.9" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 373.8" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 2.1" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 3.5" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 5.1" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 5.4" )

print( 80 * "-" )
mol.engines["mm"] = qm3.engines._qmlj.run( mol, l_qm, l_mm, exc )
mol.get_grad()
print( round( mol.func, 1 ), "/ -26.8" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 23.0" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 3.3" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 0.0" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 0.0" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 0.0" )
