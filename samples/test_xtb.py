import	numpy
import	qm3
import  qm3.engines.xtb
import  os
import  sys

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.guess_atomic_numbers()
print( mol.anum )
print( mol.chrg )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
print( sqm.sum(), smm.sum() )

mol.engines["qm"] = qm3.engines.xtb.run( mol, 1, 0, sqm, smm, sla )

mol.get_grad()
print( round( mol.func, 1 ), "/ -46100.8" )
print( mol.grad[sqm] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 566.8" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 70.4" )
