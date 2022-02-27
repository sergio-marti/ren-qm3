import	numpy
import	qm3
import  qm3.engines
import  io


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

f = io.StringIO( """%pal nprocs 2 end
qm3_charges
! qm3_job rks b3lyp def2-svp chelpg
*xyz 1 1
qm3_atoms
*
""" )
mol.engines.append( qm3.engines.qm3_orca( mol, f, sqm, smm, sla ) )
mol.engines[-1].exe = "/Users/smarti/Devel/orca/orca_4_2_1_macosx_openmpi314/orca orca.inp > orca.out"

mol.get_grad()
print( mol.func )
print( mol.grad )
