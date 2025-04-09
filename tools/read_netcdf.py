#!/usr/bin/env python3
import  numpy
import  qm3
import  scipy.io

mol = qm3.molecule()
mol.prmtop_read( open( "prmtop" ) )

with scipy.io.netcdf_file( "mdcrd", "r" ) as fd:
    xyz = fd.variables["coordinates"]
    mol.coor = xyz[ xyz.shape[0] // 2 ].copy()
    del( xyz )
