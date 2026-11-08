import  os
os.environ["OMP_NUM_THREADS"] = "1"
import  numpy
import  qm3
import  qm3.engines.pyscf
import  sys


cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
mol.guess_atomic_numbers()

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]

chkp = "pbc_dm.npz"
if( os.path.isfile( chkp ) ):
    os.remove( chkp )

opt = { "basis": "def2-svp", "conv_tol": 1.e-8, "charge": 1, "spin": 0, "method": "b3lypg",
    "memory": 8192, "grid": 2, "max_cyc": 60, "nproc": 2, "df": True,
    "gpu": "gpu" in sys.argv, "pbc": True, "rcut_ewald": 10.0, "rcut_hcore": 15.0, "chk": chkp }

mol.engines["qm"] = qm3.engines.pyscf.run( mol, opt, sqm, smm, sla )

# first call: fresh SCF
mol.get_grad()
e1 = mol.func
g1 = numpy.linalg.norm( mol.grad )
print( "call1 E = %.6f  |g| = %.4f  |gC6| = %.5f  |gH11| = %.5f"%( e1, g1,
        numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ),
        numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] )) )

mol.get_grad()
print( "call2 E = %.6f  dE = %g"%( mol.func, mol.func - e1 ) )
