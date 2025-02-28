#!/usr/bin/env python3
import  numpy
import  qm3
import  pickle


mol = qm3.molecule()
mol.psf_read( open( "oxy-cope.psf" ) )
mol.xyz_read( open( "reac.xyz" ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.guess_atomic_numbers()

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )
smm[sqm] = False
print( sqm.sum(), smm.sum() )

with open( "sele_QM.pk", "wb" ) as f:
    pickle.dump( numpy.flatnonzero( sqm ), f )

with open( "sele_MM.pk", "wb" ) as f:
    pickle.dump( numpy.flatnonzero( smm ), f )
