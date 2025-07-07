import  os
os.environ["OPENMM_CPU_THREADS"] = "1"
import  sys
import	numpy
import	qm3
import  qm3.engines.sander

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
box = numpy.array( [ 25.965, 29.928, 28.080 ] )

mol.pdb_read( open( cwd + "amber.pdb" ) )
mol.boxl = box

mol.engines["mm"] = qm3.engines.sander.run( cwd + "amber.prmtop", mol )
mol.get_grad()
print( round( mol.func, 1 ), "/ -23063.0" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 1173.8" )
