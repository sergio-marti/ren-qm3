import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.string
import  qm3.actions.minimize
import  qm3.utils.parallel
import  qm3.actions.minimize
import  qm3.actions.dynamics
import  sys
import  os

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

opar = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.pdb_read( open( cwd + "node.%02d"%( opar.node ) ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( cwd + "oxy-cope.psf" ) )
mol.guess_atomic_numbers()

_psf = openmm.app.charmmpsffile.CharmmPsfFile( cwd + "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( cwd + "oxy-cope.top", cwd + "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["ss"] = qm3.engines.string.string( mol, opar.node, 1e-7, open( cwd + "str.config" ) )
if( opar.ncpu != mol.engines["ss"].nwin ):
    print( ">> ncpu != nwin" )
    opar.stop()
    sys.exit( 1 )

mol.set_active( numpy.logical_or( sqm, smm ) )

def cstep( obj, stp ):
    global  opar
    ncrd = obj.engines["ss"].ncrd
    nwin = obj.engines["ss"].nwin
    ncr2 = ncrd * ncrd
    if( stp % 10 == 0 ):
        opar.barrier()
        if( opar.node == 0 ):
            crd = [ obj.engines["ss"].rcrd ]
            met = [ obj.engines["ss"].cmet ]
            for i in range( 1, nwin ):
                crd.append( opar.recv_r8( i, ncrd ) )
                met.append( numpy.array( opar.recv_r8( i, ncr2 ) ).reshape( ( ncrd, ncrd ) ) )
            crd = numpy.array( crd )
            rep = qm3.engines.string.distribute( crd, met )
            with open( "convergence", "at" ) as f:
                f.write( "%20.10lf\n"%( numpy.linalg.norm( crd - rep ) ) )
            obj.engines["ss"].rcrd = rep[0,:]
            for i in range( 1, nwin ):
                opar.send_r8( i, rep[i,:].ravel().tolist() )

            with open( "last.str", "wt" ) as f:
                for i in range( nwin ):
                    f.write( "".join( [ "%12.6lf"%( j ) for j in rep[i,:] ] ) + "\n" )
        else:
            opar.send_r8( 0, obj.engines["ss"].rcrd.ravel().tolist() )
            opar.send_r8( 0, obj.engines["ss"].cmet.ravel().tolist() )
            obj.engines["ss"].rcrd = numpy.array( opar.recv_r8( 0, ncrd ) )


#qm3.actions.minimize.fire( mol, print_frequency = 1, current_step = cstep, step_number = 100, gradient_tolerance = 0 )
qm3.actions.dynamics.langevin_verlet( mol, print_frequency = 1, current_step = cstep, step_number = 1000 )

mol.engines.pop( "mm" )
mol.engines.pop( "ss" )
mol.get_func()
with open( "s_node.%02d"%( opar.node ), "wt" ) as f:
    f.write( "REMARK func = %20.3lf\n"%( mol.func ) )
    mol.pdb_write( f )

opar.barrier()
opar.stop()
