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


opt = { "basis": "def2-svp",
    "conv_tol": 1.e-9,
    "charge": 1,
    "spin": 0,
    "method": "b3lypg",
    "memory": 4096, # MB
    "grid": 3,
    "max_cyc": 50,
    "nproc": 2 }
mol.engines.append( qm3.engines.qm3_pyscf( mol, opt, sqm, smm, sla ) )

mol.get_grad()
print( mol.func )
print( mol.grad )
