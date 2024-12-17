import	numpy
import	qm3
import  qm3.engines.nwchem
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
#sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]

f = io.StringIO( """start nwchem
geometry units angstroms nocenter noautoz noautosym
qm3_atoms
end
basis
 * library def2-svp
end
charge 1
dft
 qm3_guess
 mulliken
 mult 1
 iterations 100
 direct
 xc b3lyp 
end
qm3_charges
qm3_job
""" )
mol.engines["qm"] = qm3.engines.nwchem.run( mol, f, sqm, smm, sla )
with open( "r.nwchem", "wt" ) as f:
    f.write( """
export NWCHEM_BASIS_LIBRARY=./nwchem/data/libraries/
mpirun -n 4 ./nwchem/nwchem nwchem.nw >& nwchem.log
""" )

mol.get_grad()
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ -697658.1" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 573.2" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 67.3" )
else:
    print( round( mol.func, 1 ), "/ -697689.3" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 568.6" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 64.0" )
