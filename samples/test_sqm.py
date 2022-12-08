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
#sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
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
print( round( mol.func, 1 ), "/ 845.8" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 714.2" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 155.1" )
