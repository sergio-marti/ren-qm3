#!/usr/bin/env python3
import  parmed
prm = parmed.charmm.CharmmParameterSet( "charmm.top", "charmm.prm" )
print( prm.residues )
xml = parmed.openmm.OpenMMParameterSet.from_parameterset( prm )
xml.write( "prm.xml" )
