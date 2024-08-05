#!/usr/bin/env python3
import  math
import  numpy
import  struct
import  qm3
import  qm3.utils
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.mopac
import  qm3.engines.mmres
import  qm3.actions.dynamics
import  time
import  io
import  sys
import  os
import	pickle
import  zipfile


node = int( sys.argv[1] )

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop" ) )
zzz = zipfile.ZipFile( "otfs.zip", "r" )
mol.xyz_read( zzz.open( "node.%03d"%( node ), "r" ), replace = True )
zzz.close()

with open( "namd_npt.xsc" ) as f:
    l = f.readline()
    while( l[0] == "#" ):
        l = f.readline()
    tmp = [ float( s ) for s in l.split() ]
    mol.boxl = numpy.array( [ tmp[1], tmp[5], tmp[9] ] )

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

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, sqm, "CPU" )

with open( "sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 0, 1, sqm, smm, sla )

mol.chrg[sqm] = 0.0
mol.engines["mm"].update_chrg( mol )

sel = numpy.logical_or( sqm, smm )
for i,j in sla:
    sel[i] = True
    sel[j] = True
mol.set_active( sel )

kb = 5000
with open( "pmf_s.delz", "rt" ) as f:
    rx  = node * float( f.read().strip() )

mol.engines["umb"] = qm3.engines.mmres.colvar_s( mol, kb, rx,
    open( "pmf_s.cnf", "rt" ), open( "pmf_s.str", "rt" ), open( "pmf_s.met", "rt" ) )

mol.fdat = open( "dat.%03d"%( node ), "wt" )
mol.fdat.write( "%12.6lf%12.6lf\n"%( kb, rx ) )
mol.fdat.flush()
mol.fgeo = open( "geo.%03d"%( node ), "wt" )
mol.xgeo = []
with open( "pmf_s.cnf", "rt" ) as f:
    f.readline()
    for l in f:
        mol.xgeo.append( [ int( i ) for i in l.split() ] )
print( mol.xgeo )

#if( os.path.isfile( "rst.%03d"%( node ) ) ):
#    with open( "rst.%03d"%( node ), "rb" ) as f:
#        mol.coor = pickle.load( f )
#        mol.velo = pickle.load( f )
#    mol.engines["mm"].update_coor( mol )

def cstep( obj, stp ):
    global  node
    obj.fdat.write( "%12.6f\n"%( obj.rval["umb"][1] ) )
    obj.fdat.flush()
    if( stp % 1000 == 0 ):
        with open( "rst.%03d"%( node ), "wb" ) as f:
            pickle.dump( obj.coor, f )
            pickle.dump( obj.velo, f )
    for i,j in obj.xgeo:
        obj.fgeo.write( "%20.10lf"%( qm3.utils.distance( obj.coor[i], obj.coor[j] ) ) )
    obj.fgeo.write( "\n" )
    obj.fgeo.flush()

qm3.actions.dynamics.langevin_verlet( mol, step_size = 0.0005, temperature = 300.0,
    gamma_factor = 100.0, print_frequency = 100, step_number = 10000,
    current_step = cstep )

mol.fdat.close()
mol.fgeo.close()
