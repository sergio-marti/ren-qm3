#!/usr/bin/env python3
import  sys
import  numpy
import  zipfile
import  qm3.engines.string

ncrd, nwin, nstp = sys.argv[1:4]

try:
    skip = int( sys.argv[4] )
except:
    skip = 1000

z_cvs = zipfile.ZipFile( "otfs_cvs.zip", "r" )
l_cvs = [ z_cvs.open( f, "r" ) for f in z_cvs.namelist() ]

z_met = zipfile.ZipFile( "otfs_met.zip", "r" )
l_met = [ z_met.open( f, "r" ) for f in z_met.namelist() ]

z_frc = zipfile.ZipFile( "otfs_frc.zip", "r" )
l_frc = [ z_frc.open( f, "r" ) for f in z_frc.namelist() ]

qm3.engines.string.integrate_mfep( int( ncrd ), int( nwin ), int( nstp ), l_cvs, l_met, l_frc, skip )

for f in l_cvs:
    f.close()
z_cvs.close()
for f in l_met:
    f.close()
z_met.close()
for f in l_frc:
    f.close()
z_frc.close()
