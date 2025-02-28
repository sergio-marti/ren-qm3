#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.string
import  qm3.actions.dynamics
import  qm3.utils.parallel
import  pickle
import  sys
import  zipfile

par = qm3.utils.parallel.client_fsi( int( sys.argv[1] ) )

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
zzz = zipfile.ZipFile( "neb.zip", "r" )
mol.xyz_read( zzz.open( "node.%02d"%( par.node ), "r" ), replace = True )
zzz.close()

_psf = openmm.app.charmmpsffile.CharmmPsfFile( "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( "oxy-cope.top", "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

mol.engines["ss"] = qm3.engines.string.string( mol, par.node, open( "str.config" ) )

mol.x_cvs = []
mol.x_frc = []
mol.x_met = []

def cstep( obj, stp ):
    global  par

    ncrd = obj.engines["ss"].ncrd
    nwin = obj.engines["ss"].nwin
    ncr2 = ncrd * ncrd

    obj.x_cvs.append(   obj.engines["ss"].ccrd )
    obj.x_met.append(   obj.engines["ss"].cmet )
    obj.x_frc.append( - obj.engines["ss"].cdif )

    if( stp % 100 == 0 ):

        obj.engines["ss"].integrate()
        #obj.engines["ss"].integrate( 1.e-6 )

        par.barrier()

        if( par.node == 0 ):
            crd = [ obj.engines["ss"].rcrd ]
            met = [ obj.engines["ss"].amet ]
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
            par.send_r8( 0, obj.engines["ss"].amet.ravel().tolist() )
            obj.engines["ss"].rcrd = numpy.array( par.recv_r8( 0, ncrd ) )

        obj.engines["ss"].initialize_averages()


qm3.actions.dynamics.langevin_verlet( mol, print_frequency = 1, current_step = cstep,
                step_size = 0.001, step_number = 10000 )

numpy.savetxt( "node.%02d.cvs"%( par.node ), numpy.array( mol.x_cvs ) )
numpy.savetxt( "node.%02d.frc"%( par.node ), numpy.array( mol.x_frc ) )
mol.x_met = numpy.array( mol.x_met )
numpy.savetxt( "node.%02d.met"%( par.node ), mol.x_met.reshape( ( mol.x_met.shape[0], mol.x_met.shape[1] * mol.x_met.shape[2] ) ) )

with open( "node.%02d"%( par.node ), "wt" ) as f:
    mol.xyz_write( f )

par.barrier()
par.stop()
