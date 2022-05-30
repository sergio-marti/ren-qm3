import	numpy
import	qm3
import  qm3.engines.dftd4
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
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]

prm = { "chrg": 1.0, "s6": 1.00, "s8": 2.02929367, "a1": 0.40868035, "a2": 4.53807137  }
mol.engines["qm"] = qm3.engines.dftd4.run( mol, prm, sqm, sla )

mol.get_grad()
print( round( mol.func, 1 ), "/ -37.5" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 4.5" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 2.7" )
