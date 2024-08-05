#!/usr/bin/env python3
import  math
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.mopac
import  qm3.engines.string
import  qm3.engines.mmres
import  qm3.actions.dynamics
import  qm3.actions.neb
import  qm3.utils.parallel
import  os
import  pickle
import  sys
import  zipfile


par = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop" ) )
zzz = zipfile.ZipFile( "neb.zip", "r" )
mol.xyz_read( zzz.open( "node.%03d"%( par.node ), "r" ), replace = True )
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

emm = qm3.engines.openmm.run( _sys, _top, sqm, "CPU" )
mol.engines["mm"] = emm

with open( "sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

eqm = qm3.engines.mopac.run( mol, "AM1", 0, 1, sqm, smm, sla )
mol.engines["qm"] = eqm

sel = numpy.logical_or( sqm, smm )
for i,j in sla:
    sel[i] = True
    sel[j] = True
mol.set_active( sel )

mol.engines["ss"] = qm3.engines.string.string( mol, par.node, 1e-7, open( "str.config" ) )

##mol.engines["mierda"] = qm3.engines.mmres.distance( 5000, 3.0, [ 2176, 8322 ], skip_BE = 2.0 )

def cstep( obj, stp ):
    global  par
    ncrd = obj.engines["ss"].ncrd
    nwin = obj.engines["ss"].nwin
    ncr2 = ncrd * ncrd
    if( stp % 10 == 0 ):
        par.barrier()
        if( par.node == 0 ):
            crd = [ obj.engines["ss"].rcrd ]
            met = [ obj.engines["ss"].cmet ]
            for i in range( 1, nwin ):
                crd.append( par.recv_r8( i, ncrd ) )
                met.append( numpy.array( par.recv_r8( i, ncr2 ) ).reshape( ( ncrd, ncrd ) ) )
            crd = numpy.array( crd )
            rep = qm3.engines.string.distribute( crd, met )
            obj.engines["ss"].rcrd = rep[0,:]
            for i in range( 1, nwin ):
                par.send_r8( i, rep[i,:].ravel().tolist() )
            with open( "last.str", "wt" ) as f:
                for i in range( nwin ):
                    f.write( "".join( [ "%12.6lf"%( j ) for j in rep[i,:] ] ) + "\n" )
        else:
            par.send_r8( 0, obj.engines["ss"].rcrd.ravel().tolist() )
            par.send_r8( 0, obj.engines["ss"].cmet.ravel().tolist() )
            obj.engines["ss"].rcrd = numpy.array( par.recv_r8( 0, ncrd ) )


qm3.actions.dynamics.langevin_verlet( mol, print_frequency = 1, current_step = cstep,
                step_size = 0.0005, step_number = 4000 )

mol.engines.pop( "mm" )
mol.engines.pop( "ss" )
mol.get_func()
with open( "node.%03d"%( par.node ), "wt" ) as f:
    mol.xyz_write( f, comm = "func = %20.3lf\n"%( mol.func ) )

par.barrier()
par.stop()
