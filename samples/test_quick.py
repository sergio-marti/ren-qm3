import	numpy
import	qm3
import  qm3.engines.quick
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

mol.engines["qm"] = qm3.engines.quick.run( mol,
    "dft b3lyp basis=def2-svp cutoff=1.0e-10 denserms=1.0e-6 charge=1",
    sqm, smm, sla )

mol.get_grad()
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ -697658.2" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 575.3" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 80.1" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 6.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 7.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 7.3" )
else:
    print( round( mol.func, 1 ), "/ -697689.3" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 570.4" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 75.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 0.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 2.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 1.9" )
