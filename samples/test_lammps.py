import  sys
import	numpy
import	qm3
import  qm3.engines.lammps
 
mol = qm3.molecule()
box = numpy.array( [ 25.965, 29.928, 28.080 ] )

mol.pdb_read( open( "charmm.pdb" ) )
mol.boxl = box

with open( "lammps.inp", "wt" ) as f:
    f.write( """
units           real
neigh_modify    delay 2 every 1

atom_style      full
bond_style      harmonic
angle_style     charmm
dihedral_style  charmm
improper_style  harmonic

pair_style      lj/charmm/coul/long 8 12
pair_modify     mix arithmetic
kspace_style    pppm 1e-6

read_data       lammps.data

special_bonds   charmm

#group           qmatm id 1-3
#neigh_modify    exclude group qmatm qmatm

reset_timestep  0
timestep        1.
thermo_style    multi
thermo          1
""" )

mol.engines["mm"] = qm3.engines.lammps.run()
mol.get_grad()
print( round( mol.func, 1 ), "/ -22537.6" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 1857.3" )
print( mol.grad[0:19,:] )
