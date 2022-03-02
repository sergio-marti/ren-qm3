import	numpy
import	qm3
import  qm3.engines.pyscf
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


opt = { "basis": "def2-svp",
    "conv_tol": 1.e-9,
    "charge": 1,
    "spin": 0,
    "method": "b3lypg",
    "memory": 4096, # MB
    "grid": 3,
    "max_cyc": 50,
    "nproc": 2 }
mol.engines["qm"] = qm3.engines.pyscf.run( mol, opt, sqm, smm, sla )

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -697633.7524811694 ) < 1.e-4 ), "function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 575.7223290637156 ) < 1.e-4 ), "gradient error"
print( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) - 68.42149795651093 ) < 1.e-4 ), "QM-LA gradient error"
