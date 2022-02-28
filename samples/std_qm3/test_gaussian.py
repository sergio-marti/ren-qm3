import  numpy
import	qm3.mol
import  qm3.fio.xplor
import  qm3.engines.gaussian
import  io


mol = qm3.mol.molecule( "../charmm.pdb" )
qm3.fio.xplor.psf_read( mol, "../charmm.psf" )
mol.guess_atomic_numbers()
print( mol.anum[0:3], mol.anum[-3:] )
print( mol.chrg[0:3], mol.chrg[-3:] )

sqm = [ mol.indx["A"][1][a] for a in ['N1', 'C2', 'C3', 'N4', 'C5', 'H7', 'H8', 'C10', 'H16', 'H17', 'H18', 'H19'] ]
smm = mol.sph_sel( sqm, 12 )
print( len( sqm ), len( smm ) )
sla = [ [ mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ] ]
smm = list( set( smm ).difference( set( sqm + sum( sla, [] ) ) ) )

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
eng = qm3.engines.gaussian.run_single( mol, f, sqm, smm, sla )
eng.exe = ". ~/Devel/g09/pgi.imac64/g09.profile; g09 gauss.com"

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
eng.get_grad( mol )

print( mol.func )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )
