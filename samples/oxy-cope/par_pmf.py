import  os
import  sys
import  multiprocessing as mp

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

def calc( who ):
    print( who )
    os.system( "python3 " + cwd + "cvs_pmf.py %d"%( who ) )

if( __name__ == "__main__" ):
    lst = list( range( 42 ) )
    wrk = mp.Pool( processes = int( sys.argv[1] ) )
    wrk.map( calc, lst )
