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
if( len( sys.argv ) > 1 and sys.argv[1] == "single" ):
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
else:
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
print( sqm.sum(), smm.sum() )

f = io.StringIO( """%pal nprocs 4 end
qm3_charges
! qm3_job rks b3lyp def2-svp chelpg
*xyz 1 1
qm3_atoms
*
""" )
mol.engines["qm"] = qm3.engines.orca.run( mol, f, sqm, smm, sla )
mol.engines["qm"].exe = "./orca/orca orca.inp > orca.out"

mol.get_grad()
#
# orca_6.0.1 (linux)
#
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ -697208.3" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 576.5" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 81.4" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 5.9" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 7.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 7.3" )
else:
    print( round( mol.func, 1 ), "/ -697239.4" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 571.6" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 76.4" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 0.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 2.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 1.9" )
