import	numpy
import	qm3
import	qm3.data
import  qm3.engines.tmole
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
#sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]

with open( "onlyqm", "wt" ) as f:
    f.write( "%d\n\n"%( numpy.sum( sqm ) + len( sla ) ) )
    for i in numpy.flatnonzero( sqm ):
        f.write( "%2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                        mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] ) )
    if( len( sla[0] ) == 2 ):
        for i,j in sla:
            v = ( mol.coor[j] - mol.coor[i] ) / numpy.linalg.norm( mol.coor[j] - mol.coor[i] ) * 1.1
            f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H",
                        mol.coor[i,0] + v[0], mol.coor[i,1] + v[1], mol.coor[i,2] + v[2] ) )
    else:
        for i,j,k in sla:
            v = ( mol.coor[j] - mol.coor[i] ) / numpy.linalg.norm( mol.coor[j] - mol.coor[i] ) * 1.1
            f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H",
                        mol.coor[i,0] + v[0], mol.coor[i,1] + v[1], mol.coor[i,2] + v[2] ) )

os.system( """x2t onlyqm > coord
define > /dev/null << EOD

slave
a coord
*
no
b all def2-SVP
*
eht
y
1
y
ri
on
*
dft
func b3lyp-g
on
*
scf
iter
200
conv
8

*
*
EOD
sed -i -e "s/\\$drvopt/\\$drvopt\\n   point charges/" -e "s/\\$end/\\$point_charges file=charges\\n\\$point_charge_gradients file=charges.gradient\\n\\$end/" control
""" )

mol.engines["qm"] = qm3.engines.tmole.run( mol, sqm, smm, sla )

mol.get_grad()
if( len( sla[0] ) == 2 ):
    print( round( mol.func, 1 ), "/ -697682.2" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 551.7" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 75.6" )
else:
    print( round( mol.func, 1 ), "/ -697713.5" )
    print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 547.6" )
    print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 72.7" )

