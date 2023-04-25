#!/usr/bin/env python3
import  math
import  numpy
import  struct
import  qm3
import  qm3.utils
import  qm3.utils._dcd
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.gaussian
import  qm3.engines.mmres
import  qm3.actions.dynamics
import  time
import  io
import  sys
import  os
import	pickle


node = int( sys.argv[1] )

if( not os.path.isdir( "x_%02d"%( node ) ) ):
	os.mkdir( "x_%02d"%( node ) )
os.chdir( "x_%02d"%( node ) )

mol = qm3.molecule()
box = numpy.array( [ 92.154, 102.242, 97.285 ] )
mol.prmtop_read( open( "../complex.prmtop" ) )
mol.xyz_read( open( "../node.%02d"%( node ), "rt" ), replace = True )
mol.boxl = box

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "../complex.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )
_sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3( box[0], 0.0, 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, box[1], 0.0 ) * openmm.unit.angstrom,
        openmm.Vec3( 0.0, 0.0, box[2] ) * openmm.unit.angstrom )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "../sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, sqm, "CPU" )

with open( "../sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "../sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

f = io.StringIO( """%chk=gauss.chk
%mem=24000mb
%nproc=24
#p m062x/6-31+g(d,p) qm3_job qm3_guess charge prop=(field,read) scf=(conver=6,direct) nosymm fchk

.

0 1
qm3_atoms

qm3_charges

qm3_field
""" )
mol.engines["qm"] = qm3.engines.gaussian.run( mol, f, sqm, smm, sla )
mol.engines["qm"].exe = "source /gpfs/home/uji40/uji40094/g09_d01/g09.profile; g09 gauss.com"

mol.set_active( numpy.logical_or( sqm, smm ) )
for i,j in sla:
    mol.actv[j] = True

kb = 3000
with open( "../pmf_s.delz", "rt" ) as f:
    rx  = node * float( f.read().strip() )

mol.engines["umb"] = qm3.engines.mmres.colvar_s( mol, kb, rx,
    open( "../pmf_s.cnf", "rt" ), open( "../pmf_s.str", "rt" ), open( "../pmf_s.met", "rt" ) )

mol.fdat = open( "../dat.%02d"%( node ), "wt" )
mol.fdat.write( "%20.10lf%20.10lf\n"%( kb, rx ) )
mol.fdat.flush()

mol.fdcd = qm3.utils._dcd.dcd()
mol.fdcd.open_write( "../dcd.%02d"%( node ), mol.natm )

with open( "../pmf_s.cnf", "rt" ) as f:
    f.readline()
    mol.ageo = [ int( i ) for i in f.read().split() ]
mol.fgeo = open( "../geo.%02d"%( node ), "wt" )

if( os.path.isfile( "../rst.%02d"%( node ) ) ):
    with open( "../rst.%02d"%( node ), "rb" ) as f:
        mol.coor = pickle.load( f )
        mol.velo = pickle.load( f )
    mol.engines["mm"].update_coor( mol )

def cstep( obj, stp ):
    global  node
    obj.fdat.write( "%20.10lf\n"%( obj.rval["umb"][1] ) )
    obj.fdat.flush()
    with open( "../rst.%02d"%( node ), "wb" ) as f:
        pickle.dump( obj.coor, f )
        pickle.dump( obj.velo, f )
    if( stp % 50 == 0 ):
        obj.fdcd.append( obj )
    for i in range( len( obj.ageo ) // 2 ):
        obj.fgeo.write( "%12.6lf"%( qm3.utils.distance( obj.coor[obj.ageo[2*i]], obj.coor[obj.ageo[2*i+1]] ) ) )
    obj.fgeo.write( "\n" )
    obj.fgeo.flush()

qm3.actions.dynamics.langevin_verlet( mol, step_size = 0.0005, temperature = 310.0,
    gamma_factor = 100.0, print_frequency = 1, step_number = 500,
    current_step = cstep )

mol.fdat.close()
mol.fdcd.close()
mol.fgeo.close()
