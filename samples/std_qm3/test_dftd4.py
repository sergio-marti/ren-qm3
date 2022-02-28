import  numpy
import	qm3.mol
import  qm3.fio.xplor
import  qm3.engines.dftd4


mol = qm3.mol.molecule( "../charmm.pdb" )
qm3.fio.xplor.psf_read( mol, "../charmm.psf" )
mol.guess_atomic_numbers()
print( mol.anum[0:3], mol.anum[-3:] )
print( mol.chrg[0:3], mol.chrg[-3:] )

sqm = [ mol.indx["A"][1][a] for a in ['N1', 'C2', 'C3', 'N4', 'C5', 'H7', 'H8', 'C10', 'H16', 'H17', 'H18', 'H19'] ]
sla = [ [ mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ] ]

prm = { "chrg": 1.0, "s6": 1.00, "s8": 2.02929367, "a1": 0.40868035, "a2": 4.53807137  }
eng = qm3.engines.dftd4.run_dynlib( mol, prm, sqm, sla )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
eng.get_grad( mol )

print( mol.func )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )
