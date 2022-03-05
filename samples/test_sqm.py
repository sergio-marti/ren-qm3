import	numpy
import	qm3
import  qm3.engines.sqm
import  io
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

f = io.StringIO( """
_slave_
&qmmm 
maxcyc    = 0,
qm_theory = "AM1",
qmcharge  = 1,
qmmm_int  = 1,
verbosity = 4
 /
qm3_atoms
qm3_charges
""" )

mol.engines["qm"] = qm3.engines.sqm.run( mol, f, sqm, smm, sla )

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - 845.8024 ) < 0.001 ), "function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 714.1929 ) < 0.001 ), "gradient error"
print( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) - 155.1103 ) < 0.001 ), "QM-LA gradient error"
