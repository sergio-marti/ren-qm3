import	numpy
import	qm3
import  qm3.engines.mmres
import  os
import  sys

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )

sel = [ mol.indx["A"][1][a] for a in [ "C3", "N4", "C5", "C10" ] ]

print( ">> distance" )
mol.engines["res"] = qm3.engines.mmres.distance( 2000., 1.6,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"] ] )
mol.get_grad()
print( mol.rval["res"][1], round( mol.func, 1 ), "/ 8.2" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 255.5" )

print( "\n>> angle" )
mol.engines["res"] = qm3.engines.mmres.angle( 2000., 115,
    [ mol.indx["A"][1]["C3"], mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"] ] )
mol.get_grad()
print( mol.rval["res"][1], round( mol.func, 1 ), "/ 35.7" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 596.8" )

print( "\n>> dihedral" )
mol.engines["res"] = qm3.engines.mmres.dihedral( { 3: [ 0.8159, 0.0 ] },
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] )
mol.get_grad()
print( mol.rval["res"][1], round( mol.func, 1), "/ 1.6" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 3 ), "/ 0.016" )

print( "\n>> improper" )
mol.engines["res"] = qm3.engines.mmres.improper( 2000., 0.0,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] )
mol.get_grad()
print( mol.rval["res"][1], round( mol.func, 4 ), "/ 0.0013" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 4.2" )

print( "\n>> multiple_distance" )
mol.engines["res"] = qm3.engines.mmres.multiple_distance( 2000., -0.2,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C5"] ],
    numpy.array( [ -1.0, 1.0 ] ) )
mol.get_grad()
print( mol.rval["res"][1], round( mol.func, 1 ), "/ 3.5" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 271.3" )

print( "\n>> tether" )
mol.engines["res"] = qm3.engines.mmres.tether( mol, 2000., mol.resn == "SUS" )
mol.coor *= 0.95
mol.get_grad()
print( round( mol.func, 1 ), "/ 356.5" )
print( mol.grad[sel] )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 1194.1" )
