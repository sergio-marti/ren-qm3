import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.utils
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.actions.minimize
import  qm3.utils.parallel
import  sys
import  os
import  random
import  pickle
import  time


cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep
con = qm3.utils.parallel.client_mpi()

mol = qm3.molecule()
mol.pdb_read( open( cwd + "reac.pdb" ) )
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

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "OpenCL" )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

a1  = mol.indx["A"][1]["C2"]
a2  = mol.indx["A"][1]["C3"]
a3  = mol.indx["A"][1]["C5"]
a4  = mol.indx["A"][1]["C6"]
rx1 = [ 1.4, 3.5 ] 
dx1 = 0.1
rx2 = [ 1.4, 3.5 ]
dx2 = 0.1
nx1 = int( ( rx1[1] - rx1[0] ) / dx1 ) + 1
nx2 = int( ( rx2[1] - rx2[0] ) / dx2 ) + 1

random.seed()
if( con.node == 0 ):
    print( "rnge:", nx1, nx2 )
    print( "ncpu:", con.ncpu )
    ci = int( round( ( qm3.utils.distance( mol.coor[a1], mol.coor[a2] ) - rx1[0] ) / dx1, 0 ) )
    ci = min( max( 0, ci ), nx1 - 1 )
    cj = int( round( ( qm3.utils.distance( mol.coor[a3], mol.coor[a4] ) - rx2[0] ) / dx2, 0 ) )
    cj = min( max( 0, cj ), nx2 - 1 )
    mol.engines["r1"] = qm3.engines.mmres.distance( 5000, rx1[0] + dx1 * ci, [ a1, a2 ] )
    mol.engines["r2"] = qm3.engines.mmres.distance( 5000, rx2[0] + dx2 * cj, [ a3, a4 ] )
    qm3.actions.minimize.fire( mol, print_frequency = 100, gradient_tolerance = 0.1, step_number = 2000,
            current_step = lambda obj,stp: sys.stdout.flush() )
    mol.pdb_write( open( "pes.%02d.%02d.pdb"%( ci, cj ), "wt" ) )
    mol.engines.pop( "r1" ); mol.engines.pop( "r2" )
    # ----------------- distribute
    flg = []
    par = []
    idx = []
    n   = nx1 * nx2
    for i in range( nx1 ):
        for j in range( nx2 ):
            flg.append( False )
            par.append( (i,j) )
            idx.append( i*nx2+j )
    flg[ci*nx2+cj] = True
    mov = [ (-1,0), (+1,0), (0,-1), (0,+1) ]
    for i in range( max( n // 100, 10 ) ):
        random.shuffle( mov )
        random.shuffle( idx )
    lst = []
    for j in range( 1, n ):
        random.shuffle( idx )
        q = True
        i = 0
        while( q and i < n ):
            if( not flg[idx[i]] ):
                random.shuffle( mov )
                k = 0
                while( q and k < 4 ):
                    I = par[idx[i]][0] + mov[k][0]
                    J = par[idx[i]][1] + mov[k][1]
                    w = I * nx2 + J
                    if( I >= 0 and I < nx1 and J >= 0 and J < nx2 and flg[w] ):
                        q = False
                        lst.append( ( par[idx[i]][0], par[idx[i]][1], "pes.%02d.%02d.pdb"%( I, J ) ) )
                        flg[idx[i]] = True
                    k += 1
            i += 1
    tsk = [ [] for i in range( con.ncpu ) ]
    for i in range( n - 1 ):
        tsk[i%con.ncpu].append( lst[i] )
    f = open( "tasks.pk", "wb" )
    pickle.dump( tsk, f )
    f.close()
    # ----------------------------
    con.barrier()
else:
    con.barrier()
    f = open( "tasks.pk", "rb" )
    tsk = pickle.load( f )
    f.close()


tmp = qm3.molecule()
for i,j,s in tsk[con.node]:
    while( not os.path.isfile( s ) ):
        time.sleep( 1 )
    time.sleep( 2 )
    tmp.pdb_read( open( s, "rt" ) )
    mol.coor = tmp.coor
    mol.engines["r1"] = qm3.engines.mmres.distance( 5000, rx1[0] + dx1 * i, [ a1, a2 ] )
    mol.engines["r2"] = qm3.engines.mmres.distance( 5000, rx2[0] + dx2 * j, [ a3, a4 ] )
    qm3.actions.minimize.fire( mol, print_frequency = 100, gradient_tolerance = 0.1, step_number = 1000,
            current_step = lambda obj,stp: sys.stdout.flush() )
    mol.pdb_write( open( "pes.%02d.%02d.pdb"%( i, j ), "wt" ) )
    mol.engines.pop( "r1" ); mol.engines.pop( "r2" )

con.barrier()
con.stop()
