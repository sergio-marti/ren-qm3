import	numpy
import	qm3
import  qm3.engines.orca
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

f = io.StringIO( """%pal nprocs 2 end
qm3_charges
! qm3_job rks b3lyp def2-svp chelpg
*xyz 1 1
qm3_atoms
*
""" )
mol.engines["qm"] = qm3.engines.orca.run( mol, f, sqm, smm, sla )
mol.engines["qm"].exe = "/Users/smarti/Devel/orca/orca_4_2_1_macosx_openmpi314/orca orca.inp > orca.out"

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -697207.4365 ) < 0.001 ), "function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 575.700 ) < 0.01 ), "gradient error"
print( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) - 68.4438 ) < 0.001 ), "QM-LA gradient error"
