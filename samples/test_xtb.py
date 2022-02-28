import	numpy
import	qm3
import  qm3.engines


mol = qm3.molecule()
mol.pdb_read( open( "charmm.pdb" ) )
mol.psf_read( open( "charmm.psf" ) )
mol.guess_atomic_numbers()
print( mol.anum )
print( mol.chrg )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
print( sqm.sum(), smm.sum() )

mol.engines.append( qm3.engines.qm3_xtb( mol, 1, 0, sqm, smm, sla ) )

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -46091.943203479954 ) < 1.e-6 ), "xTB: function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 589.1365457514921 ) < 1.e-6 ), "xTB: gradient error"
