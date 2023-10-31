import  sys
import  math
import  numpy
import  typing
import  qm3.data


numpy.random.seed()


def fake_cs( self: object, step: int ):
    pass


def current_temperature( mol: object, ndeg: int ) -> ( float, float ):
    kine = numpy.sum( mol.mass * numpy.square( mol.velo ) )
    temp = kine * 10.0 / ( ndeg * qm3.data.KB * qm3.data.NA )
    kine *= 0.005
    return( temp, kine )


def assign_velocities( mol: object, temperature: float, proj: numpy.array, ndeg: int ):
    sd = numpy.sqrt( qm3.data.KB * temperature * 1000.0 * qm3.data.NA / mol.mass )
    vx = numpy.random.normal( 0.0, sd )
    vy = numpy.random.normal( 0.0, sd )
    vz = numpy.random.normal( 0.0, sd )
    mol.velo = numpy.column_stack( ( vx, vy, vz ) ) * mol.actv.astype( numpy.float64 ) * 0.01
    mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
    cur, kin = current_temperature( mol, ndeg )
    mol.velo *= math.sqrt( temperature / cur )


def get_projector( mol:object )  -> ( int, numpy.array ):
    actv = mol.actv.sum()
    if( actv < mol.natm ):
        ndeg = 3 * actv
        proj = numpy.zeros( ( mol.natm, 1 ) )
    else:
        ndeg = 3 * actv - 3
        proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass * mol.actv.astype( numpy.float64 ) ) ) * mol.actv.astype( numpy.float64 )
    return( ndeg, proj )


# =================================================================================================

def langevin_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        gamma_factor: typing.Optional[float] = 50.0, 
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    dynamics_langevin_verlet.F90 fDynamo module
    Chem. Phys. v236, p243 (1998) [doi:10.1016/S0301-0104(98)00214-6]
    """
    log_file.write( "---------------------------------------- Dynamics: Langevin-Verlet (NVT)\n\n" )
    ndeg = 3 * mol.actv.sum()
    if( mol.actv.sum() < mol.natm ):
        proj = numpy.zeros( ( mol.natm, 1 ) )
        log_file.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    else:
        ndeg -= 3
        proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass * mol.actv.astype( numpy.float64 ) ) ) * mol.actv.astype( numpy.float64 )
        log_file.write( "Degrees of Freedom: %20ld [removing COM]\n"%( ndeg ) )
    log_file.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
    log_file.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
    log_file.write( "Gamma Factor:       %20.10lg (ps^-1)\n"%( gamma_factor ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    ff  = step_size * gamma_factor
    if( ff < 0.01 ):
        ff = 0.01
        log_file.write( "\n>> Gamma factor:    %20.10lg (ps^-1)"%( 0.01 / step_size ) )
    log_file.write( "\n%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)", "Temperature (K)" ) )
    log_file.write( 100 * "-" + "\n" )
    c0   = math.exp( - ff )
    c1   = ( 1.0 - c0 ) / ff
    c2   = ( 1.0 - c1 ) / ff
    sr   = step_size * math.sqrt( ( 2.0 - ( 3.0 - 4.0 * c0 + c0 * c0 ) / ff ) / ff )
    sv   = math.sqrt( 1.0 - c0 * c0 )
    cv1  = step_size * ( 1.0 - c0 ) * ( 1.0 - c0 ) / ( ff * sr * sv )
    cv2  = math.sqrt( 1.0 - cv1 * cv1 )
    fr1  = step_size * c1
    fv1  = step_size * ( c1 - c2 )
    fv2  = step_size * c2
    fr2  = step_size * fv2
    sdev = 0.01 * numpy.sqrt( qm3.data.KB * temperature * 1000.0 * qm3.data.NA / mol.mass ) * mol.actv.astype( numpy.float64 )
    if( not hasattr( mol, "velo" ) ):
        assign_velocities( mol, temperature, proj, ndeg )
    temp, kine = current_temperature( mol, ndeg )
    mol.get_grad()
    cacc = - mol.grad / mol.mass * 100.0
    cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        mol.coor += fr1 * mol.velo + fr2 * cacc
        r1 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) ) * mol.actv.astype( numpy.float64 )
        r2 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) ) * mol.actv.astype( numpy.float64 )
        mol.coor += sdev * sr * r1
        oacc = c0 * mol.velo + fv1 * cacc + sdev * sv * ( cv1 * r1 + cv2 * r2 )
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo = oacc + fv2 * cacc
#        mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
        temp, kine = current_temperature( mol, ndeg )
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

# =================================================================================================

def csvr_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        temperature_coupling: typing.Optional[float] = 0.1, 
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    J. Chem. Phys. v126, p14101 (2007) [doi:10.1063/1.2408420]
    """
    log_file.write( "---------------------------------------- Dynamics: CSVR-Verlet (NVT)\n\n" )
    ndeg = 3 * mol.actv.sum()
    if( mol.actv.sum() < mol.natm ):
        proj = numpy.zeros( ( mol.natm, 1 ) )
        log_file.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    else:
        ndeg -= 3
        proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass * mol.actv.astype( numpy.float64 ) ) ) * mol.actv.astype( numpy.float64 )
        log_file.write( "Degrees of Freedom: %20ld [removing COM]\n"%( ndeg ) )
    log_file.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
    log_file.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
    log_file.write( "Temp. coupling:     %20.10lg (ps)\n"%( temperature_coupling ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "\n%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)", "Temperature (K)" ) )
    log_file.write( 100 * "-" + "\n" )
    c0   = math.exp( - step_size / temperature_coupling )
    c1   = 5.e-4 * qm3.data.KB * temperature * qm3.data.NA * ( 1.0 - c0 )
    c2   = 2.0 * math.sqrt( c0 * c1 )
    fc   = step_size
    fv   = fc * 0.5
    fa   = fc * fv
    ndeg -= 3
    if( not hasattr( mol, "velo" ) ):
        assign_velocities( mol, temperature, proj, ndeg )
    temp, kine = current_temperature( mol, ndeg )
    mol.get_grad()
    cacc = - mol.grad / mol.mass * 100.0
    cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        mol.coor += fc * mol.velo + fa * cacc
        mol.velo += fv * cacc
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo += fv * cacc
        mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
        temp, kine = current_temperature( mol, ndeg )
        # Canonical Sampling through Velocity Rescaling
        rr  = numpy.random.normal( 0.0, 1.0, ndeg )
        scv = math.sqrt( c0 + c1 / kine * ( rr[0] * rr[0] + numpy.sum( numpy.square( rr[1:] ) ) ) + c2 / math.sqrt( kine ) * rr[0] )
        # Berendsen thermostat
        #scv = math.sqrt( 1.0 + step_size / temperature_coupling * ( temperature / temp - 1.0 ) )
        mol.velo *= scv
        temp, kine = current_temperature( mol, ndeg )
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
