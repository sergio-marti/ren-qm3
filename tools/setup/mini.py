#!/usr/bin/env python3
import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.mmres
import  qm3.engines.mopac
import  qm3.actions.minimize
import  pickle

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop" ) )
mol.xyz_read( open( "guess" ) )
with open( "namd_npt.xsc" ) as f:
    l = f.readline()
    while( l[0] == "#" ):
        l = f.readline()
    tmp = [ float( s ) for s in l.split() ]
    mol.boxl = numpy.array( [ tmp[1], tmp[5], tmp[9] ] )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True
with open( "sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )
smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "complex.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )
_sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3( mol.boxl[0], 0.0, 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, mol.boxl[1], 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, 0.0, mol.boxl[2] ) * openmm.unit.angstrom )

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, sel_QM = sqm, platform = "CPU" )
mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 0, 1, sel_QM = sqm, sel_MM = smm, link = sla )

v = mol.coor[3007] - mol.coor[4545]
mol.coor[3008] = mol.coor[4545] + v / numpy.linalg.norm( v ) * 1.0
mol.engines["u" + str( len( mol.engines ) )] = qm3.engines.mmres.distance( 5000, 1.4, [ 3007, 8791 ] )
mol.engines["u" + str( len( mol.engines ) )] = qm3.engines.mmres.distance( 5000, 1.0, [ 3008, 4545 ] )

qm3.actions.minimize.fire( mol, gradient_tolerance = 8.0, print_frequency = 1, use_maxgrad = True )

mol.xyz_write( open( "temp", "wt" ) )

print( mol.engines )
for x in list( mol.engines.keys() ):
    if( x[0] == "u" ):
        mol.engines.pop( x )
print( mol.engines )

qm3.actions.minimize.fire( mol, print_frequency = 1 )

mol.xyz_write( open( "mini.xyz", "wt" ) )
