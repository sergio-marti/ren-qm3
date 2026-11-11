import  os
os.environ["OMP_NUM_THREADS"] = "1"
import  sys
import  time
import  random
import  qm3.utils.parallel

random.seed()

if( "mpi" in sys.argv ):
    meth = "MPI"
    opar = qm3.utils.parallel.client_mpi()
else:
    meth = "FSI"
    opar = qm3.utils.parallel.client_fsi( int( sys.argv[1] ) )

if( opar.node == 0 ):
    print( f"[master:{meth}]", opar.ncpu )
    for w in range( 1, opar.ncpu ):
        opar.send_r8( w, [ 355. / 113. ] )
else:
    time.sleep( 4 * random.random() )
    r = opar.recv_r8( 0, 1 )
    print( f"[node:{opar.node}]", r )

opar.barrier()
opar.stop()
