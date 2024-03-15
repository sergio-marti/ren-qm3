import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.utils
import  qm3.utils.hessian
import  qm3.actions.minimize
import  qm3.utils._mpi
import  sys
import  os

who, cpu = qm3.utils._mpi.init()

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
mol.pdb_read( open( "node.25" ) )
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
print( sqm.sum(), smm.sum(), end = " " )
smm = numpy.logical_and( smm, numpy.logical_not( sqm ) )
print( smm.sum() )

mol.set_active( sqm )

tsk = [ [] for i in range( cpu ) ]
siz = 0
for i in numpy.argwhere( sqm ).ravel():
    for j in [0, 1, 2]:
        tsk[siz%cpu].append( ( siz, i, j ) )
        siz += 1
dsp = 1.0e-4

if( who == 0 ):
    print( [ len( tsk[i] ) for i in range( cpu ) ] )
    sys.stdout.flush()

    log = open( "borra_log.mm", "wt" )

    emm = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
    mol.engines["mm"] = emm

    eqm = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
    mol.engines["qm"] = eqm

    def calc_hess( self: object, step: int ):
        sys.stdout.flush()
        # optimize MM counterpart previous a hessian calculation...
        eqm.get_func( self )
        self.set_active( smm )
        self.engines["mm"].update_chrg( self )
        self.engines.pop( "qm" )
        qm3.actions.minimize.fire( self, gradient_tolerance = 0.5, log_file = log )
        log.flush()
        self.chrg[sqm] = 0.0
        self.engines["mm"].update_chrg( self )
        self.engines["qm"] = eqm
        self.set_active( sqm )
        # -------------------------------------------------------------
        if( step % 10 == 0 ):
            crd = self.coor.ravel().tolist()
            qm3.utils._mpi.barrier()
            for i in range( 1, cpu ):
                qm3.utils._mpi.send_r8( i, crd )
            hes = numpy.zeros( ( siz, siz ), dtype=numpy.float64 )
            for k,i,j in tsk[who]:
                bak = self.coor[i,j]
                self.coor[i,j] = bak + dsp
                self.get_grad()
                gp = self.grad[sqm].ravel()
                self.coor[i,j] = bak - dsp
                self.get_grad()
                hes[k,:] = ( gp - self.grad[sqm].ravel() ) / ( 2.0 * dsp )
                self.coor[i,j] = bak

            qm3.utils._mpi.barrier()
            for i in range( 1, cpu ):
                hes += numpy.array( qm3.utils._mpi.recv_r8( i, siz * siz ) ).reshape( ( siz, siz ) )
            self.hess = 0.5 * ( hes + hes.T )

            qm3.utils.hessian.manage( self, self.hess )
            self.get_grad()
        else:
            self.get_grad()
            qm3.utils.hessian.manage( self, self.hess, should_update = True )
        return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) )

    qm3.actions.minimize.baker( mol,
        calc_hess,
        gradient_tolerance = 2.0,
        step_number = 100,
        print_frequency = 1,
        follow_mode = 0 )

    with open( "saddle.pdb", "wt" ) as f:
        mol.pdb_write( f )

    val, vec = qm3.utils.hessian.frequencies( mol, mol.hess )
    print( val[0:10] )
    qm3.utils.hessian.normal_mode( mol, val, vec, 0, afac = 8.0 )

else:
    os.mkdir( "par.%02d"%( who ) )
    os.chdir( "par.%02d"%( who ) )

    mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "OpenCL" )
    mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

    while( True ):
        qm3.utils._mpi.barrier()
        mol.coor = numpy.array( qm3.utils._mpi.recv_r8( 0, 3 * mol.natm ) ).reshape( ( mol.natm, 3 ) )

        hes = numpy.zeros( ( siz, siz ), dtype=numpy.float64 )
        for k,i,j in tsk[who]:
            bak = mol.coor[i,j]
            mol.coor[i,j] = bak + dsp
            mol.get_grad()
            gp = mol.grad[sqm].ravel()
            mol.coor[i,j] = bak - dsp
            mol.get_grad()
            hes[k,:] = ( gp - mol.grad[sqm].ravel() ) / ( 2.0 * dsp )
            mol.coor[i,j] = bak
        qm3.utils._mpi.barrier()
        qm3.utils._mpi.send_r8( 0, hes.ravel().tolist() )


qm3.utils._mpi.stop()
