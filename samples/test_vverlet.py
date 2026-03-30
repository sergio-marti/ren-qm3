import  typing
import  math
import  os
os.environ["OPENMM_CPU_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.actions.dynamics
import  qm3.utils._dcd

 

def velocity_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        scale_frequency: typing.Optional[int] = 100,
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = qm3.actions.dynamics.fake_cs ):
    """
    dynamics_velocity_verlet.F90 fDynamo module
    Chem. Phys. v236, p243 (1998) [doi:10.1016/S0301-0104(98)00214-6]
    """
    log_file.write( "---------------------------------------- Dynamics: Velocity-Verlet (NVT)\n\n" )
    ndeg = 3 * mol.actv.sum()
    if( mol.actv.sum() < mol.natm ):
        rcom = False
        proj = numpy.zeros( ( mol.natm, 1 ) )
        log_file.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    else:
        ndeg -= 3
        rcom = True
        proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass ) )
        log_file.write( "Degrees of Freedom: %20ld [removing COM]\n"%( ndeg ) )
    log_file.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
    log_file.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Scale Frequency:    %20d\n"%( scale_frequency ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "\n%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)", "Temperature (K)" ) )
    log_file.write( 100 * "-" + "\n" )
    fc = step_size
    fv = 0.5 * fc
    fa = fv  * fc
    if( not hasattr( mol, "velo" ) ):
        qm3.actions.dynamics.assign_velocities( mol, temperature, proj, ndeg )
    temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
    mol.get_grad()
    cacc = - mol.grad / mol.mass * 100.0
    if( rcom ):
        cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        mol.coor += fc * mol.velo + fa * cacc
        #oacc = mol.velo + fv * cacc
        mol.velo += fv * cacc
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        if( rcom ):
            cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        #mol.velo = oacc + fv * cacc
        mol.velo +=  fv * cacc
        if( rcom ):
            mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
        temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
        if( scale_frequency > 0 and istp % scale_frequency == 0 ):
            mol.velo *= math.sqrt( temperature / temp )
            temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
        xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
        xavr += xtmp
        xrms += numpy.square( xtmp )
        if( istp % print_frequency == 0 ):
            log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
        current_step( mol, istp )
    xavr /= step_number + 1
    xrms /= step_number + 1
    xrms = numpy.sqrt( numpy.fabs( xrms - xavr * xavr ) )
    log_file.write( 100 * "-" + "\n" )
    log_file.write( "%-20s"%( "Averages:" ) + "".join( [ "%20.5lf"%( i ) for i in xavr ] ) + "\n" )
    log_file.write( "%-20s"%( "RMS Deviations:" ) + "".join( [ "%20.5lf"%( i ) for i in xrms ] ) + "\n" )
    log_file.write( 100 * "-" + "\n" )




cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()

mol.pdb_read( open( cwd + "amber.pdb" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
mol.prmtop_read( open( cwd + "amber.prmtop" ) )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( cwd + "amber.prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 12.0 * openmm.unit.angstrom,
    switchDistance = 10.0 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )
_sys.setDefaultPeriodicBoxVectors(
    openmm.Vec3( mol.boxl[0], 0.0, 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, mol.boxl[1], 0.0 ) * openmm.unit.angstrom,
    openmm.Vec3( 0.0, 0.0, mol.boxl[2] ) * openmm.unit.angstrom )

sqm = mol.resn == "SUS"
smm = mol.sph_sel( sqm, 10 )
mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, sel_QM = sqm )
mol.engines["qm"] = qm3.engines.xtb.run( mol, 1, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["qm"].img = True

mol.get_grad()
print( mol.func )

mol.dcd = qm3.utils._dcd.dcd()
mol.dcd.open_write( "borra.dcd", mol.natm )

def cstep( self, step ):
    self.set_active( sqm )
    self.wrap()
    self.set_active()
    if( step % 10 == 0 ):
        self.dcd.append( self )

velocity_verlet( mol, step_number = 10_000, scale_frequency = 100, print_frequency = 50 , current_step = cstep )

mol.dcd.close()
