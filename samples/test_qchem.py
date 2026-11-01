import	numpy
import	qm3
import  qm3.engines.qchem
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
if( len( sys.argv ) > 1 and sys.argv[1] == "single" ):
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
else:
    sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
print( sqm.sum(), smm.sum() )

f = io.StringIO( """$rem
jobtype qm3_job
method b3lyp
basis def2-svp
qm3_guess
qm_mm true
skip_charge_self_interact true
esp_efield 2
symmetry off
sym_ignore true
print_input false
qmmm_print true
$end

$molecule
1 1
qm3_atoms
$end

qm3_charges
""" )
mol.engines["qm"] = qm3.engines.qchem.run( mol, f, sqm, smm, sla )
with open( "r.qchem", "wt" ) as f:
    f.write( """source /usr/local/chem/qchem_6.2/rc
export QCSCRATCH=`pwd`
rm -rf qchem.tmp.0
qchem -nt 10 qchem.inp qchem.log qchem.tmp > /dev/null
""" )

mol.get_grad()
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ -697633.8" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 577.9" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 82.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 5.9" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 7.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 7.3" )
else:
    print( round( mol.func, 1 ), "/ -697664.9" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 573.1" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 79.6" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C6"]] ), 1 ), "/ 0.6" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H11"]] ), 1 ), "/ 2.0" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["H12"]] ), 1 ), "/ 1.9" )
