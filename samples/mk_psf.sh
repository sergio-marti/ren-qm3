#!/bin/bash

cat > inp << EOD
topology charmm.top
segment A {
	first none
	last none
	auto angles dihedrals
	pdb charmm.pdb
}
writepsf charmm.psf
EOD
#~/Devel/namd/2.14/psfgen < inp


cat > inp << EOD
*

prnl 6
wrnl 6
bomblvl -1

open read form unit 10 name charmm.top
read rtf card unit 10
close unit 10

open read form unit 10 name charmm.prm
read parameter card unit 10
close unit 10

read sequence card
*
*
SUS WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT  -
WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT WAT 

generate A setup noangle nodihe

open unit 10 write form name charmm.psf
write psf card unit 10
close unit 10
EOD
~/Devel/charmm/43b1/charmm < inp
