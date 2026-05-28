import	numpy
import	qm3
import  qm3.data
import  qm3.engines.gaussian
import  qm3.utils.eint
import  io
import  sys
import  os
import  re


cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.guess_atomic_numbers()

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]

f = io.StringIO( """%chk=gauss.chk
%mem=2048mb
%nproc=2
#p b3lyp/def2svp qm3_job qm3_guess charge scf=direct nosymm fchk

.

1 1
qm3_atoms

qm3_charges
""" )
mol.engines["qm"] = qm3.engines.gaussian.run( mol, f, sqm, smm, sla )
mol.engines["qm"].exe = ". ./g09.profile; g09 gauss.com"

mol.get_func()

if( not os.path.isfile( "dens.cube" ) ):
    os.system( ". ./g09.profile; formchk gauss.chk; cubegen 0 Density=SCF gauss.fchk gauss.cube 0 h" )

with open( "0scf.com", "wt" ) as f:
    f.write( """%chk=gauss.chk
%mem=2048mb
%nproc=2
#p b3lyp/def2svp guess=read geom=check scf=(qc,direct,maxcyc=1,conver=-4) nosymm

.

1 1


""" )
os.system( ". ./g09.profile; g09 0scf.com" )
with open( "0scf.log", "rt" ) as f:
    ref = float( re.compile( r"SCF Done:[^\.]+=[\ ]+([\-\.0-9]+)" ).findall( f.read() )[0] ) * qm3.data.H2J


smm[sqm] = False
chg = mol.chrg.copy()
for i,j,k in sla:
    smm[j] = False
    for l in k:
        chg[l] += mol.chrg[j] / len( k )


print( "with charges:", mol.func )
print( "gas phase (0scf):", ref )
print( "eint:", mol.func - ref )

rho = qm3.utils.eint.read_rho_cube( open( "gauss.cube", "rt" ), +1, True )
print( "rho x mm_chrg:", qm3.utils.eint.calc_rho_eint( rho, mol.coor[smm], chg[smm] ) )
