#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.utils.parallel
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.minimize
import  qm3.actions.neb
import  sys
import  pickle

opar = qm3.utils.parallel.client_fsi( int( sys.argv[1] ) )

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.xyz_read( open( "reac.xyz" ) )
mol.guess_atomic_numbers()

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

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )

eqm = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

reac = mol.coor.copy()

mol.xyz_read( open( "prod.xyz" ) )
prod = mol.coor.copy()

gues = qm3.actions.neb.distribute( 40, [ reac[sqm], prod[sqm] ] )
if( opar.node == 0 ):
    print( "gues:", len( gues ) )

opar.barrier()

if( opar.node == 0 ):
    def backup( obj, stp ):
        g = numpy.round( numpy.linalg.norm( obj.grad ) / numpy.sqrt( obj.natm * 3 ), 2 )
        if( g < 4 ):
            os.system( "tar -cf o-%.2lf.tar node.??"%( g ) )
        sys.stdout.flush()

    obj = qm3.actions.neb.neb( gues, 200, opar )
    qm3.actions.minimize.fire( obj, step_number = 500, print_frequency = 1, gradient_tolerance = 0.5, current_step = backup )
    opar.barrier()
    for who in range( 1, opar.ncpu ):
        opar.send_i4( who, [ 0 ] )

else:
    geom = []
    for i in range( len( gues ) // 2 ):
        geom.append( reac.copy() )
        geom[-1][sqm] = gues[i].copy()
    for i in range( len( gues ) // 2, len( gues ) ):
        geom.append( prod.copy() )
        geom[-1][sqm] = gues[i].copy()

    flog = open( "slave_%03d.log"%( opar.node ), "wt" )

    sele = numpy.flatnonzero( sqm )
    dime = len( sele )
    opar.barrier()
    nchk = opar.recv_i4( 0, 1 )[0]
    chnk = opar.recv_i4( 0, nchk )
    print( ">>", opar.node, nchk, chnk )
    size = dime * nchk
    grad = numpy.zeros( ( size, 3 ) )
    opar.barrier()
    flag = opar.recv_i4( 0, 1 )[0]
    while( flag ):
        # get current coordinates for my chunks
        coor = numpy.array( opar.recv_r8( 0, 3 * size ) )
        coor.shape = ( size, 3 )
        # calculate gradients
        vpot = []
        for who in range( nchk ):
            ii = who * dime
            mol.coor = geom[chnk[who]].copy()
            mol.coor[sele] = coor[ii:ii+dime]
            # optimize environment
            mol.set_active( smm )
            qm3.actions.minimize.fire( mol, log_file = flog,
                                       print_frequency = 1,
                                       current_step = lambda obj, stp: flog.flush() )
            # get gradient
            mol.set_active( sqm )
            mol.func = 0.0
            mol.grad = numpy.zeros( ( mol.natm, 3 ) )
            eqm.get_grad( mol )
            mol.project_gRT()
            vpot.append( mol.func )
            grad[ii:ii+dime] = mol.grad[sele]
            # store coordinates
            with open( "node.%02d"%( chnk[who] ), "wt" ) as f:
                mol.xyz_write( f, comm = "func: %.4lf"%( mol.func ) )
            geom[chnk[who]] = mol.coor.copy()
        # send functions and gradients
        opar.barrier()
        opar.send_r8( 0, vpot )
        opar.send_r8( 0, grad.ravel().tolist() )
        # wait for more
        opar.barrier()
        flag = opar.recv_i4( 0, 1 )[0]

opar.stop()
