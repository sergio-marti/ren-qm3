import  sys
import  numpy
import  typing
import  qm3.data


numpy.random.seed()

# =================================================================================================

def steepest_descent( mol: object,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 15.,
        fdesc: typing.Optional = sys.stdout ):
    fdesc.write( "---------------------------------------- Minimization (SD)\n\n" )
    ndf = 3 * mol.actv.sum()
    fdesc.write( "Degrees of Freedom: %20ld\n"%( ndf ) )
    fdesc.write( "Step Number:        %20d\n"%( step_number ) )
    fdesc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdesc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdesc.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    ndf = numpy.sqrt( ndf )
    mol.get_grad()
    norm = numpy.linalg.norm( mol.grad )
    if( norm > step_size ):
        ssiz = step_size
    elif( norm > gradient_tolerance ):
        ssiz = norm
    else:
        ssiz = gradient_tolerance
    grms = norm / ndf
    fdesc.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    fdesc.write( "-" * 70 + "\n" )
    fdesc.write( "%10s%20.5lf%20.8lf%20.10lf\n"%( "", mol.func, grms, ssiz ) )
    i = 0
    while( i < step_number and grms > gradient_tolerance ):
        mol.coor -= mol.grad / norm * ssiz
        mol.get_grad()
        norm = numpy.linalg.norm( mol.grad )
        if( norm > step_size ):
            ssiz = step_size
        elif( norm > gradient_tolerance ):
            ssiz = norm
        else:
            ssiz = gradient_tolerance
        grms = norm / ndf
        i = i + 1
        if( i%print_frequency == 0 ):
            fdesc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( i, mol.func, grms, ssiz ) )
        mol.current_step( i )
    if( i%print_frequency != 0 ):
        fdesc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( i + 1, mol.func, grms, ssiz ) )
    fdesc.write( "-" * 70 + "\n\n" )

# =================================================================================================

def fire( mol: object,
        step_number: typing.Optional[int] = 1000,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        mixing_alpha: typing.Optional[float] = 0.1,
        delay_step: typing.Optional[int] = 5,
        exit_uphill: typing.Optional[bool] = False,
        fdesc: typing.Optional = sys.stdout ):
    fdesc.write( "---------------------------------------- Minimization (FIRE)\n\n" )
    ndeg = 3 * mol.actv.sum()
    fdesc.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    fdesc.write( "Step Number:        %20d\n"%( step_number ) )
    fdesc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdesc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdesc.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    fdesc.write( "Checking UpHill:    %20s\n"%( exit_uphill ) )
    fdesc.write( "Mixing Alpha:       %20.10lg\n"%( mixing_alpha ) )
    fdesc.write( "Delay Step:         %20d\n"%( delay_step ) )
    fdesc.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    fdesc.write( "-" * 70 + "\n" )
    ndeg = numpy.sqrt( ndeg )
    nstp = 0
    ssiz = step_size
    alph = mixing_alpha
    velo = numpy.zeros( ( mol.natm, 3 ) )
    mol.get_grad()
    qfun = True
    norm = numpy.linalg.norm( mol.grad )
    grms = norm / ndeg
    fdesc.write( "%10s%20.5lf%20.10lf\n"%( "", mol.func, grms ) )
    i = 0
    while( i < step_number and grms > gradient_tolerance and qfun ):
        if( - numpy.sum( velo * mol.grad ) > 0.0 ):
            vsiz = numpy.linalg.norm( velo )
            velo = ( 1.0 - alph ) * velo - alph * mol.grad / norm * vsiz
            if( nstp > delay_step ):
                ssiz = min( ssiz * 1.1, step_size )
                alph *= 0.99
            nstp += 1
        else:
            alph = mixing_alpha
            ssiz *= 0.5
            nstp = 0
            velo = numpy.zeros( ( mol.natm, 3 ) )
        velo -= ssiz * mol.grad
        step = ssiz * velo
        tmp  = numpy.linalg.norm( step )
        if( tmp > ssiz ):
            step *= ssiz / tmp
        mol.coor += step

        lfun = mol.func
        mol.get_grad()
        norm = numpy.linalg.norm( mol.grad )
        grms = norm / ndeg
        if( exit_uphill ):
            if( lfun < mol.func ):
                fdesc.write( ">> search become uphill!\n" )
                qfun = False
                mol.coor -= step

        i = i + 1
        if( i%print_frequency == 0 ):
            fdesc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( i, mol.func, grms, ssiz ) )
        mol.current_step( i )
    if( i%print_frequency != 0 ):
        fdesc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( i + 1, mol.func, grms, ssiz ) )
    fdesc.write( "-" * 70 + "\n\n" )

# =================================================================================================

def langevin_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        gamma_factor: typing.Optional[float] = 50.0, 
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        fdesc: typing.Optional = sys.stdout ):

    def __current_temperature( mol: object, mass: numpy.array, ndeg: int ) -> ( float, float ):
        kine = numpy.sum( mass * numpy.square( mol.velo ) )
        temp = kine * 10.0 / ( ndeg * qm3.data.KB * qm3.data.NA )
        kine *= 0.005
        return( temp, kine )

    def __assign_velocities( mol: object, temperature: float, mass: numpy.array, proj: numpy.array, ndeg: int ):
        sd = numpy.sqrt( qm3.data.KB * temperature * 1000.0 * qm3.data.NA / mass )
        vx = numpy.random.normal( 0.0, sd )
        vy = numpy.random.normal( 0.0, sd )
        vz = numpy.random.normal( 0.0, sd )
        mol.velo = numpy.column_stack( ( vx, vy, vz ) ) * mol.actv * 0.01
        mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
        cur, kin = __current_temperature( mol, mass, ndeg )
        mol.velo *= numpy.sqrt( temperature / cur )

    fdesc.write( "---------------------------------------- Dynamics: Langevin-Verlet (NVT)\n\n" )
    ndeg = 3 * mol.actv.sum()
    fdesc.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    fdesc.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
    fdesc.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
    fdesc.write( "Gamma Factor:       %20.10lg (ps^-1)\n"%( gamma_factor ) )
    fesc.write( "Step Number:        %20d\n"%( step_number ) )
    fdesc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    ff  = step_size * gamma_factor
    if( ff < 0.01 ):
        ff = 0.01
        fdesc.write( "\n>> Gamma factor:    %20.10lg (ps^-1)"%( 0.01 / step_size ) )
    fdesc.write( "\n%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)", "Temperature (K)" ) )
    fdesc.write( 100 * "-" + "\n" )
    c0   = numpy.exp( - ff )
    c1   = ( 1.0 - c0 ) / ff
    c2   = ( 1.0 - c1 ) / ff
    sr   = step_size * numpy.sqrt( ( 2.0 - ( 3.0 - 4.0 * c0 + c0 * c0 ) / ff ) / ff )
    sv   = numpy.sqrt( 1.0 - c0 * c0 )
    cv1  = step_size * ( 1.0 - c0 ) * ( 1.0 - c0 ) / ( ff * sr * sv )
    cv2  = numpy.sqrt( 1.0 - cv1 * cv1 )
    fr1  = step_size * c1
    fv1  = step_size * ( c1 - c2 )
    fv2  = step_size * c2
    fr2  = step_size * fv2
    mass = mol.mass.reshape( ( mol.natm, 1 ) )
    sdev = 0.01 * numpy.sqrt( qm3.data.KB * temperature * 1000.0 * qm3.data.NA / mass ) * mol.actv
    ndeg -= 3
    proj = ( numpy.sqrt( mass / numpy.sum( mass * mol.actv ) ) * mol.actv )
    if( not hasattr( mol, "velo" ) ):
        __assign_velocities( mol, temperature, mass, proj, ndeg )
    temp, kine = __current_temperature( mol, mass, ndeg )
    mol.get_grad()
    cacc = - mol.grad / mass * 100.0
    cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ] )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    fdesc.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        mol.coor += fr1 * mol.velo + fr2 * cacc
        r1 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) )
        r2 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) )
        mol.coor += sdev * sr * r1
        oacc = c0 * mol.velo + fv1 * cacc + sdev * sv * ( cv1 * r1 + cv2 * r2 )
        mol.get_grad()
        cacc = - mol.grad / mass * 100.0
        cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo = oacc + fv2 * cacc
        mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
        temp, kine = __current_temperature( mol, mass, ndeg )
        xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ] )
        xavr += xtmp
        xrms += numpy.square( xtmp )
        if( istp % print_frequency == 0 ):
            fdesc.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3] ) )
        mol.current_step( istp )
    xavr /= step_number + 1
    xrms /= step_number + 1
    xrms = numpy.sqrt( numpy.fabs( xrms - xavr * xavr ) )
    fdesc.write( 100 * "-" + "\n" )
    fdesc.write( "%-20s"%( "Averages:" ) + "".join( [ "%20.5lf"%( i ) for i in xavr ] ) + "\n" )
    fdesc.write( "%-20s"%( "RMS Deviations:" ) + "".join( [ "%20.5lf"%( i ) for i in xrms ] ) + "\n" )
    fdesc.write( 100 * "-" + "\n" )
