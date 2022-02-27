cat > acs.pdb << EOD
ATOM      1  N1  SUS     1      -0.226   0.308   0.498  1.00  0.00      SUS 
ATOM      2  C2  SUS     1       0.681   1.022   1.273  1.00  0.00      SUS 
ATOM      3  C3  SUS     1      -0.139  -1.055   0.811  1.00  0.00      SUS 
ATOM      4  N4  SUS     1       1.288   0.054   2.033  1.00  0.00      SUS 
ATOM      5  C5  SUS     1       0.824  -1.207   1.784  1.00  0.00      SUS 
ATOM      6  C6  SUS     1       0.438  -2.806   3.705  1.00  0.00      SUS 
ATOM      7  H7  SUS     1       2.311  -2.496   2.647  1.00  0.00      SUS 
ATOM      8  H8  SUS     1       1.092  -3.336   1.703  1.00  0.00      SUS 
ATOM      9  C9  SUS     1       0.851  -4.144   4.356  1.00  0.00      SUS 
ATOM     10  C10 SUS     1       1.237  -2.512   2.416  1.00  0.00      SUS 
ATOM     11  H11 SUS     1      -0.636  -2.834   3.476  1.00  0.00      SUS 
ATOM     12  H12 SUS     1       0.587  -1.993   4.429  1.00  0.00      SUS 
ATOM     13  H13 SUS     1       0.267  -4.326   5.267  1.00  0.00      SUS 
ATOM     14  H14 SUS     1       1.913  -4.138   4.632  1.00  0.00      SUS 
ATOM     15  H15 SUS     1       0.678  -4.986   3.672  1.00  0.00      SUS 
ATOM     16  H16 SUS     1       0.865   2.096   1.277  1.00  0.00      SUS 
ATOM     17  H17 SUS     1      -0.758  -1.810   0.332  1.00  0.00      SUS 
ATOM     18  H18 SUS     1       2.011   0.243   2.719  1.00  0.00      SUS 
ATOM     19  H19 SUS     1      -0.857   0.711  -0.190  1.00  0.00      SUS 
END
EOD

source ~/Devel/amber/rc

antechamber -i acs.pdb -fi pdb -fo ac -o acs.ac -c bcc -at gaff -pf y -nc 1
prepgen -i acs.ac -o acs.prepin -rn SUS
parmchk2 -i acs.prepin -f prepi -o acs.frcmod -a Y -s gaff

cat > leap.in << EOD
source leaprc.gaff
source oldff/leaprc.ff03
loadamberparams frcmod.ions1lm_126_tip3p
loadamberparams frcmod.ions234lm_126_tip3p
loadamberparams acs.frcmod
loadamberprep acs.prepin
complex = loadpdb acs.pdb
solvatebox complex TIP3PBOX 10.0 0.8
saveamberparm complex amber.prmtop borra.inpcrd
savepdb complex amber.pdb
quit
EOD
tleap -f leap.in
