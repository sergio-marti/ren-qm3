import	numpy
import	qm3
import  qm3.engines.plumed
import  os
import  sys
import  io

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )

sel = [ mol.indx["A"][1][a] for a in [ "C5", "C10" ] ]

os.system( "rm -f colvar plumed.*" )

with open( "plumed.dat", "wt" ) as f:
    f.write( """UNITS LENGTH=A
dst: DISTANCE ATOMS=5,10 NOPBC
rst: RESTRAINT ARG=dst AT=1.6 KAPPA=2000.0
PRINT ARG=dst,rst.bias FILE=colvar STRIDE=1
""" )

mol.engines["res"] = qm3.engines.plumed.run( mol )
mol.get_grad()
print( round( mol.func, 1 ), "/ 8.2" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 255.5" )

for i in range( 10 ):
    mol.get_grad()

mol.engines["res"].stop()
