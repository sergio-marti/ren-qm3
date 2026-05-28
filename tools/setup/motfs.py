#!/usr/bin/env python3
import  numpy
import  qm3
import  qm3.engines.sander
import  qm3.engines.string
import  qm3.actions.dynamics
import  qm3.utils._dcd
import  qm3.utils.parallel
import  os
import  zipfile

par = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.prmtop_read( open( "complex.prmtop" ) )
zzz = zipfile.ZipFile( "neb.zip", "r" )
mol.xyz_read( zzz.open( "node.%03d"%( par.node ), "r" ), replace = True )
zzz.close()

with open( "mini_i.xyz", "rt" ) as f:
    f.readline()
    mol.boxl = numpy.array( [ float( i ) for i in f.readline().split() ] )

mol.engines["qmmm"] = qm3.engines.sander.run( "complex.prmtop", mol,
        qm_mask = ":629 | (:312 & ! @CA,HA,N,H,C,O) | (:206 | (:205 & @C,O) | (:207 & @N,H,CA,HA))",
        qm_meth = "xTB", qm_cut = 24 )

mol.engines["ss"] = qm3.engines.string.string( mol, par.node, open( "str.config" ), True )


def cstep( obj, stp ):
    global  par, WHO

    ncrd = obj.engines["ss"].ncrd
    nwin = obj.engines["ss"].nwin
    ncr2 = ncrd * ncrd

    obj.x_cvs.append(   obj.engines["ss"].ccrd )
    obj.x_met.append(   obj.engines["ss"].cmet )
    obj.x_frc.append( - obj.engines["ss"].cdif )

    if( stp % 100 == 0 ):

        if( WHO == "s6" ):
            obj.dcd.append( obj )

        obj.engines["ss"].integrate()

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
            with open( WHO + "/last.str", "wt" ) as f:
                for i in range( nwin ):
                    f.write( "".join( [ "%12.6lf"%( j ) for j in rep[i,:] ] ) + "\n" )

        else:
            par.send_r8( 0, obj.engines["ss"].rcrd.ravel().tolist() )
            par.send_r8( 0, obj.engines["ss"].amet.ravel().tolist() )
            obj.engines["ss"].rcrd = numpy.array( par.recv_r8( 0, ncrd ) )

        obj.engines["ss"].initialize_averages()


for WHO in [ "s1", "s2", "s3", "s4", "s5", "s6" ]:

    if( par.node == 0 ):
        try:
            os.mkdir( WHO )
        except:
            pass

    if( WHO == "s6" ):
        mol.dcd = qm3.utils._dcd.dcd()
        mol.dcd.open_write( WHO + "/dcd.%03d"%( par.node ), mol.natm )
        mol.dcd.append( mol )

    mol.x_cvs = []
    mol.x_frc = []
    mol.x_met = []

    qm3.actions.dynamics.langevin_verlet( mol, print_frequency = 1, current_step = cstep,
                step_size = 0.0005, step_number = 10_000 )

    numpy.savetxt( WHO + "/node.%03d.cvs"%( par.node ), numpy.array( mol.x_cvs ) )
    numpy.savetxt( WHO + "/node.%03d.frc"%( par.node ), numpy.array( mol.x_frc ) )
    mol.x_met = numpy.array( mol.x_met )
    numpy.savetxt( WHO + "/node.%03d.met"%( par.node ), mol.x_met.reshape( ( mol.x_met.shape[0], mol.x_met.shape[1] * mol.x_met.shape[2] ) ) )

    with open( WHO + "/node.%03d"%( par.node ), "wt" ) as f:
        mol.xyz_write( f )

    if( WHO == "s6" ):
        mol.dcd.close()

    par.barrier()

par.stop()
