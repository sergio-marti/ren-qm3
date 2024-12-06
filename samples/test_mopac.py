import  os
os.environ["OMP_NUM_THREADS"] = "1"
import	numpy
import	qm3
import  qm3.engines.mopac
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

mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 1, 1, sqm, smm, sla )

mol.get_grad()
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ 874.4" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 714.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 154.8" )
else:
    print( round( mol.func, 1 ), "/ 816.2" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 712.3" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 152.7" )
