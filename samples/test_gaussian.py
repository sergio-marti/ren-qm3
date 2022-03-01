import	numpy
import	qm3
import  qm3.engines.gaussian
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
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]

f = io.StringIO( """%chk=gauss.chk
%mem=2048mb
%nproc=2
#p b3lyp/def2svp qm3_job qm3_guess charge prop=(field,read) scf=direct nosymm fchk

.

1 1
qm3_atoms

qm3_charges

qm3_field
""" )
mol.engines.append( qm3.engines.gaussian.run( mol, f, sqm, smm, sla ) )
mol.engines[-1].exe = ". ~/Devel/g09/pgi.imac64/g09.profile; g09 gauss.com"

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -697633.7376989075 ) < 1.e-4 ), "function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 575.7344091968865 ) < 1.e-4 ), "gradient error"
print( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ) - 68.42685642072145 ) < 1.e-4 ), "QM-LA gradient error"
