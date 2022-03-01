import  numpy
import	qm3.mol
import  qm3.engines.mmres

mol = qm3.mol.molecule( "../charmm.pdb" )

sel = [ mol.indx["A"][1][a] for a in [ "C3", "N4", "C5", "C10" ] ]

def calculate( tag, eng ):
    mol.func = 0.0
    mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
    rval = eng.get_grad( mol )
    print( tag )
    print( mol.func, rval )
    grd = numpy.array( mol.grad ).reshape( ( mol.natm, 3 ) )
    print( grd[sel] )
    print( numpy.linalg.norm( grd ) )

calculate ( ">> distance", qm3.engines.mmres.distance( 2000., 1.6,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"] ] ) )

calculate( "\n>> angle", qm3.engines.mmres.angle( 2000., 115,
    [ mol.indx["A"][1]["C3"], mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"] ] ) )

calculate( "\n>> dihedral", qm3.engines.mmres.dihedral( { 3: [ 0.8159, 0.0 ] },
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] ) )

calculate( "\n>> improper", qm3.engines.mmres.improper( 2000., 0.0,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C3"] ] ) )

calculate( "\n>> multiple_distance", qm3.engines.mmres.multiple_distance( 2000., -0.2,
    [ mol.indx["A"][1]["C5"], mol.indx["A"][1]["C10"],
        mol.indx["A"][1]["N4"], mol.indx["A"][1]["C5"] ], [ -1.0, 1.0 ] ) )


eng = qm3.engines.mmres.tether( mol, 2000., list( mol.indx["A"][1].values() ) )
mol.coor = [ i * 0.95 for i in mol.coor ]
mol.func = 0.0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
eng.get_grad( mol )
print( "\n>> tether" )
print( mol.func )
grd = numpy.array( mol.grad ).reshape( ( mol.natm, 3 ) )
print( grd[sel] )
print( numpy.linalg.norm( grd ) )
