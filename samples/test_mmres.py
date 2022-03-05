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
print( mol.func, mol.rval[0] )
assert( numpy.fabs( mol.func - 8.1575 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 255.4604 ) < 0.001 ), "gradient error"

print( "\n>> angle" )
mol.engines["res"] = qm3.engines.mmres.angle( 2000., 115,
    [ mol.indx["A"][1]["C3"], mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"] ] )
mol.get_grad()
print( mol.func, mol.rval[0] )
assert( numpy.fabs( mol.func - 35.7225 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 596.7694 ) < 0.001 ), "gradient error"

print( "\n>> dihedral" )
mol.engines["res"] = qm3.engines.mmres.dihedral( { 3: [ 0.8159, 0.0 ] },
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] )
mol.get_grad()
print( mol.func, mol.rval[0] )
assert( numpy.fabs( mol.func - 1.6318 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 0.0156 ) < 0.001 ), "gradient error"

print( "\n>> improper" )
mol.engines["res"] = qm3.engines.mmres.improper( 2000., 0.0,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] )
mol.get_grad()
print( mol.func, mol.rval[0] )
assert( numpy.fabs( mol.func - 0.0013 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 4.2457 ) < 0.001 ), "gradient error"

print( "\n>> multiple_distance" )
mol.engines["res"] = qm3.engines.mmres.multiple_distance( 2000., -0.2,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C5"] ],
    numpy.array( [ -1.0, 1.0 ] ) )
mol.get_grad()
print( mol.func, mol.rval[0] )
assert( numpy.fabs( mol.func - 3.5035 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 271.3023 ) < 0.001 ), "gradient error"

print( "\n>> tether" )
mol.engines["res"] = qm3.engines.mmres.tether( mol, 2000., mol.resn == "SUS" )
mol.coor *= 0.95
mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - 356.4791 ) < 0.001 ), "function error"
print( mol.grad[sel] )
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 1194.1174 ) < 0.001 ), "gradient error"
