#!/bin/bash

prt=enzyme.pdb
lig=ligand.mol2

source ~/Devel/amber/rc
source ~/Devel/openbabel/rc
vina=~/Devel/docking/AutoDock-Vina/build/mac/release/vina

obabel -imol2 $lig -opdbqt -partialcharges eem | grep -E -v "^USER|^TER" > lig.x

rm -f borra.*
cat > inp << EOD
source leaprc.gaff
source oldff/leaprc.ff03
source leaprc.water.tip3p
prt = loadpdb $prt
saveamberparm prt borra.prmtop borra.tmp.inpcrd
quit
EOD
tleap -f inp

cpptraj << EOD
parm borra.prmtop
trajin borra.tmp.inpcrd
center origin
principal dorotation
trajout borra.inpcrd
EOD

ambpdb -p borra.prmtop -c borra.inpcrd -pqr > borra.pqr

obabel -ipqr borra.pqr -opdbqt | grep -E "^ATOM|^HETATM" > prt.x


python3 << EOD
import  numpy as np

coor = []
f = open( "borra.pqr", "rt" )
for l in f:
    if( l[0:4] == "ATOM" ):
        t = l.split()
        coor.append( [ float( t[-6] ), float( t[-5] ), float( t[-4] ) ] )
f.close()
coor = np.array( coor, dtype = float )

def d_indx( w ):
    t = w // 2
    o = list( range( - t, t + 1 ) )
    if( w % 2 == 0 ):
        del o[o.index( 0 )]
    return( o )

cent = ( np.max( coor, axis = 0 ) - np.min( coor, axis = 0 ) ) // 30 + 1
print( cent )
c = 0
for i in d_indx( int( cent[0] ) ):
    for j in d_indx( int( cent[1] ) ):
        for k in d_indx( int( cent[2] ) ):
            f = open( "inp_%04d"%( c ), "wt" )
            f.write( """receptor = prt.x
ligand = lig.x
center_x = %.1lf
center_y = %.1lf
center_z = %.1lf
size_x = 30.
size_y = 30.
size_z = 30.
energy_range = 10
num_modes = 20
out = out_%04d
"""%( 15.0 * i, 15.0 * j, 15.0 * k, c ) )
            c += 1
EOD

echo "load prt.x, format=pdbqt" > view.pml
rm -f view.vmd
for ff in inp_????; do
    gg=`echo $ff | cut -c5-`
    $vina --cpu 4 --config $ff | tee log_$gg
    grep -E "MODEL|ATOM|ENDMDL" out_$gg > out_$gg.pdb  
    cat >> view.vmd << EOD
mol new out_$gg.pdb type pdb first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol delrep 0 top
mol representation Licorice 0.100000 12.000000 12.000000
mol color Name
mol selection {all}
mol material Opaque
mol addrep top
EOD
    echo "load out_$gg.pdb, format=pdb" >> view.pml
done

cat >> view.vmd << EOD
mol new prt.x type pdb first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol delrep 0 top
mol representation Surf 1.400000 0.000000
mol color Name
mol selection {all}
mol material Opaque
mol addrep top
EOD

cat >> view.pml << EOD
util.cba( 144, "all", _self=cmd )
util.cba(  33, "prt", _self=cmd )
cmd.hide( "cartoon", "prt" )
cmd.show( "surface", "prt" )
set all_states, on
cmd.zoom( "all", animate=-1 )
clip atoms, 5, all
EOD
