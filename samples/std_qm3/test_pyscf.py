import  numpy
import	qm3.mol
import  qm3.fio.xplor
import  qm3.engines.pyscf


mol = qm3.mol.molecule( "../charmm.pdb" )
qm3.fio.xplor.psf_read( mol, "../charmm.psf" )
mol.guess_atomic_numbers()
print( mol.anum[0:3], mol.anum[-3:] )
print( mol.chrg[0:3], mol.chrg[-3:] )

sqm = [ mol.indx["A"][1][a] for a in ['N1', 'C2', 'C3', 'N4', 'C5', 'H7', 'H8', 'C10', 'H16', 'H17', 'H18', 'H19'] ]
smm = mol.sph_sel( sqm, 12 )
print( len( sqm ), len( smm ) )
sla = [ [ mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ] ]
smm = list( set( smm ).difference( set( sqm + sum( sla, [] ) ) ) )

opt = { "basis": "def2-svp",
    "conv_tol": 1.e-9,
    "charge": 1,
    "spin": 0,
    "method": "b3lypg",
    "memory": 4096, # MB
    "grid": 3,
    "max_cyc": 50,
    "nproc": 2 }
eng = qm3.engines.pyscf.run_native( mol, opt, sqm, smm, sla )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
eng.get_grad( mol )

print( mol.func )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )
w = mol.indx["A"][1]["C10"]
print( numpy.linalg.norm( numpy.array( mol.grad[3*w:3*w+3] ) ) )
