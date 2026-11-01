import  sys
import  math
import  numpy
import  typing
import  qm3.data


numpy.random.seed()


def fake_cs( self: object, step: int ):
    sys.stdout.flush()


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
        #proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass * mol.actv.astype( numpy.float64 ) ) ) * mol.actv.astype( numpy.float64 )
        proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass ) )[:,None]
    return( ndeg, proj )


# =================================================================================================

def langevin_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        gamma_factor: typing.Optional[float] = 25.0, 
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
    if( rcom ):
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
        if( rcom ):
            cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo = oacc + fv2 * cacc
#        if( rcom ):
#            mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
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
    if( not hasattr( mol, "velo" ) ):
        assign_velocities( mol, temperature, proj, ndeg )
    temp, kine = current_temperature( mol, ndeg )
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
        mol.velo += fv * cacc
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        if( rcom ):
            cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo += fv * cacc
        temp, kine = current_temperature( mol, ndeg )
        # Canonical Sampling through Velocity Rescaling
        rr  = numpy.random.normal( 0.0, 1.0, ndeg )
        scv = math.sqrt( c0 + c1 / kine * ( rr[0] * rr[0] + numpy.sum( numpy.square( rr[1:] ) ) ) + c2 / math.sqrt( kine ) * rr[0] )
        # Berendsen thermostat
        #scv = math.sqrt( 1.0 + step_size / temperature_coupling * ( temperature / temp - 1.0 ) )
        mol.velo *= scv
#        if( rcom ):
#            mol.velo -= numpy.sum( mol.velo * proj, axis = 0 ) * proj
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

class nose_hoover_chain( object ):
    """
    Nosé-Hoover extended variables (chain of chain_length masses) coupled to the
    active degrees of freedom of the system, in the Martyna-Tuckerman-Tobias-Klein
    formulation (Mol. Phys. v87, p1117 (1996)).

    - temperature    : target temperature (K)
    - damping_time   : thermostat relaxation time Tclav (ps); the chain masses are
                       Q_0 = ndeg kB T Tclav^2 and Q_k = kB T Tclav^2, k > 0
    - chain_length   : number of chain variables (1 -> original Nosé-Hoover thermostat,
                       not recommended for small/harmonic systems, lack of ergodicity)
    - chain_loops    : number of sub-steps used to propagate the chain within each
                       half MD step (improves accuracy at a small cost)
    """
    def __init__( self, ndeg: int, temperature: float, damping_time: float,
                        chain_length: typing.Optional[int] = 3,
                        chain_loops: typing.Optional[int] = 1 ):
        if( ndeg <= 0 ):
            raise ValueError( "number of degrees of freedom must be positive" )
        if( temperature <= 0.0 ):
            raise ValueError( "temperature must be positive" )
        if( damping_time <= 0.0 ):
            raise ValueError( "damping time must be positive" )
        if( chain_length < 1 ):
            raise ValueError( "chain length must be a positive integer" )
        if( chain_loops < 1 ):
            raise ValueError( "number of chain loops must be a positive integer" )
        self.ndeg        = ndeg
        self.temperature = temperature
        self.damping     = damping_time
        self.chain       = chain_length
        self.loops       = chain_loops
        self.kbt         = 0.001 * qm3.data.KB * qm3.data.NA * temperature       # kJ/mol
        self.ke_target   = ndeg * self.kbt                                       # kJ/mol
        self.eta         = numpy.zeros( chain_length, dtype = numpy.float64 )
        self.etadot      = numpy.zeros( chain_length + 1, dtype = numpy.float64 ) # the last element is a zero dummy
        self.etadotdot   = numpy.zeros( chain_length, dtype = numpy.float64 )
        self.qmass       = self.kbt * damping_time * damping_time * numpy.ones( chain_length, dtype = numpy.float64 )
        self.qmass[0]   *= ndeg
        for k in range( 1, chain_length ):
            self.etadotdot[k] = - self.kbt / self.qmass[k]


    def kinetic_temperature( self, mol: object ) -> ( float, float ):
        return( current_temperature( mol, self.ndeg ) )


    def double_kinetic( self, mol: object ) -> float:
        """
        2KE in kJ/mol: with m [amu] and v [Å/ps] the conversion factor is 1/100
        (this is the same physical convention implicit in current_temperature()
        and in the Maxwell distribution of assign_velocities(); the kinetic energy
        printed in the dynamics logs follows the internal 5/1000 convention
        already used by the other actions of this module)
        """
        return( 0.01 * numpy.sum( mol.mass * numpy.square( mol.velo ) ) )


    def integrate( self, mol: object, dthalf: float ):
        """
        propagates the chain over a thermostat half-step of length dthalf and applies
        the corresponding scaling to the (active) atomic velocities
        """
        dts = dthalf / self.loops
        self.etadotdot[0] = ( self.double_kinetic( mol ) - self.ke_target ) / self.qmass[0]
        for i in range( self.loops ):
            # -- chain variables, innermost to outermost (1st half)
            for k in range( self.chain - 1, 0, -1 ):
                c = math.exp( - 0.25 * dts * self.etadot[k+1] )
                self.etadot[k] *= c
                self.etadot[k] += 0.5 * dts * self.etadotdot[k]
                self.etadot[k] *= c
            # -- outermost variable (1st half; etadot has one extra zero dummy element)
            c = math.exp( - 0.25 * dts * self.etadot[1] )
            self.etadot[0] *= c
            self.etadot[0] += 0.5 * dts * self.etadotdot[0]
            self.etadot[0] *= c
            # -- scaling of the atomic velocities and refresh of the kinetic energy
            f = math.exp( - dts * self.etadot[0] )
            mol.velo *= numpy.where( mol.actv, f, 1.0 )
            self.etadotdot[0] = ( self.double_kinetic( mol ) - self.ke_target ) / self.qmass[0]
            # -- heat-bath coordinates
            self.eta += dts * self.etadot[:self.chain]
            # -- outermost variable (2nd half)
            self.etadot[0] *= c
            self.etadot[0] += 0.5 * dts * self.etadotdot[0]
            self.etadot[0] *= c
            # -- chain variables, outermost to innermost (2nd half)
            for k in range( 1, self.chain ):
                c = math.exp( - 0.25 * dts * self.etadot[k+1] )
                self.etadot[k] *= c
                self.etadotdot[k] = ( self.qmass[k-1] * self.etadot[k-1] * self.etadot[k-1] - self.kbt ) / self.qmass[k]
                self.etadot[k] += 0.5 * dts * self.etadotdot[k]
                self.etadot[k] *= c
        return( current_temperature( mol, self.ndeg ) )


    def energy( self ) -> float:
        """
        heat-bath contribution to the conserved quantity
        (eq. (2) @ Mol. Phys. v87, p1117 (1996)):

            kT ke_target eta_0 + 1/2 Q_0 etadot_0^2 + Sum(k=1.., kT eta_k + 1/2 Q_k etadot_k^2)
        """
        ebs = self.ke_target * self.eta[0] + 0.5 * self.qmass[0] * self.etadot[0] * self.etadot[0]
        for k in range( 1, self.chain ):
            ebs += self.kbt * self.eta[k] + 0.5 * self.qmass[k] * self.etadot[k] * self.etadot[k]
        return( ebs )


def nose_hoover_verlet( mol: object,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        damping_time: typing.Optional[float] = 0.1,
        chain_length: typing.Optional[int] = 3,
        chain_loops: typing.Optional[int] = 1,
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    deterministic NVT dynamics via Nosé-Hoover chains (Martyna, Tuckerman, Tobias, Klein,
    Mol. Phys. v87, p1117 (1996)), second-order symmetric factorisation of the equations
    of motion (Tuckerman, Berne, Martyna, J. Chem. Phys. v110, p7119 (1999)):

        exp(L_NH dt/2) exp(L_v dt/2) exp(L_x dt) exp(L_v dt/2) exp(L_NH dt/2)

    the conserved quantity is potential + kinetic + heat-bath energies, which is
    monitored in the last two columns of the log
    """
    log_file.write( "---------------------------------------- Dynamics: Nose-Hoover-Chain Verlet (NVT)\n\n" )
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
    log_file.write( "Damping Time:       %20.10lg (ps)\n"%( damping_time ) )
    log_file.write( "Chain Length:       %20d\n"%( chain_length ) )
    log_file.write( "Chain Loops:        %20d\n"%( chain_loops ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    sep  = 140 * "-"
    log_file.write( "\n%20s%20s%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)",
                "Temperature (K)", "Heat Bath (kJ/mol)", "Conserved (kJ/mol)" ) )
    log_file.write( sep + "\n" )
    nhc  = nose_hoover_chain( ndeg, temperature, damping_time, chain_length, chain_loops )
    if( not hasattr( mol, "velo" ) ):
        assign_velocities( mol, temperature, proj, ndeg )
    temp, kine = nhc.kinetic_temperature( mol )
    mol.get_grad()
    cacc = - mol.grad / mol.mass * 100.0
    if( rcom ):
        cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
    ebs  = nhc.energy()
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp, ebs, mol.func + kine + ebs ], dtype=numpy.float64 )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3], xtmp[4], xtmp[5] ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        nhc.integrate( mol, 0.5 * step_size )
        mol.velo += 0.5 * step_size * cacc
        mol.coor += step_size * mol.velo
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        if( rcom ):
            cacc -= numpy.sum( cacc * proj, axis = 0 ) * proj
        mol.velo += 0.5 * step_size * cacc
        nhc.integrate( mol, 0.5 * step_size )
        temp, kine = nhc.kinetic_temperature( mol )
        ebs  = nhc.energy()
        xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp, ebs, mol.func + kine + ebs ], dtype=numpy.float64 )
        xavr += xtmp
        xrms += numpy.square( xtmp )
        if( istp % print_frequency == 0 ):
            log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3], xtmp[4], xtmp[5] ) )
        current_step( mol, istp )
    xavr /= step_number + 1
    xrms /= step_number + 1
    xrms = numpy.sqrt( numpy.fabs( xrms - xavr * xavr ) )
    log_file.write( sep + "\n" )
    log_file.write( "%-20s"%( "Averages:" ) + "".join( [ "%20.5lf"%( i ) for i in xavr ] ) + "\n" )
    log_file.write( "%-20s"%( "RMS Deviations:" ) + "".join( [ "%20.5lf"%( i ) for i in xrms ] ) + "\n" )
    log_file.write( sep + "\n" )

# =================================================================================================

class stepper__langevin_verlet( object ):
    def __init__( self: object, mol: object,
                    step_size: typing.Optional[float] = 0.001,
                    temperature: typing.Optional[float] = 300.0,
                    gamma_factor: typing.Optional[float] = 50.0, 
                    print_frequency: typing.Optional[int] = 100,
                    log_file: typing.Optional[typing.IO] = sys.stdout ):

        self.log_file = log_file
        self.step_size = step_size
        self.temperature = temperature
        self.print_frequency = print_frequency

        self.log_file.write( "-------------------------------- Stepper Dynamics: Langevin-Verlet (NVT)\n\n" )

        self.ndeg = 3 * mol.actv.sum()
        if( mol.actv.sum() < mol.natm ):
            self.rcom = False
            self.proj = numpy.zeros( ( mol.natm, 1 ) )
            self.log_file.write( "Degrees of Freedom: %20ld\n"%( self.ndeg ) )
        else:
            self.ndeg -= 3
            self.rcom = True
            self.proj = numpy.sqrt( mol.mass / numpy.sum( mol.mass ) )
            self.log_file.write( "Degrees of Freedom: %20ld [removing COM]\n"%( self.ndeg ) )

        self.log_file.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
        self.log_file.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
        self.log_file.write( "Gamma Factor:       %20.10lg (ps^-1)\n"%( gamma_factor ) )
        self.log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )

        ff  = step_size * gamma_factor
        if( ff < 0.01 ):
            ff = 0.01
            self.log_file.write( "\n>> Gamma factor:    %20.10lg (ps^-1)"%( 0.01 / self.step_size ) )

        self.log_file.write( "\n%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)", "Total (kJ/mol)", "Temperature (K)" ) )
        self.log_file.write( 100 * "-" + "\n" )

        self.c0   = math.exp( - ff )
        self.c1   = ( 1.0 - self.c0 ) / ff
        self.c2   = ( 1.0 - self.c1 ) / ff
        self.sr   = step_size * math.sqrt( ( 2.0 - ( 3.0 - 4.0 * self.c0 + self.c0 * self.c0 ) / ff ) / ff )
        self.sv   = math.sqrt( 1.0 - self.c0 * self.c0 )
        self.cv1  = step_size * ( 1.0 - self.c0 ) * ( 1.0 - self.c0 ) / ( ff * self.sr * self.sv )
        self.cv2  = math.sqrt( 1.0 - self.cv1 * self.cv1 )
        self.fr1  = step_size * self.c1
        self.fv1  = step_size * ( self.c1 - self.c2 )
        self.fv2  = step_size * self.c2
        self.fr2  = step_size * self.fv2
        self.sdev = 0.01 * numpy.sqrt( qm3.data.KB * temperature * 1000.0 * qm3.data.NA / mol.mass ) * mol.actv.astype( numpy.float64 )
        if( not hasattr( mol, "velo" ) ):
            assign_velocities( mol, temperature, self.proj, self.ndeg )
        temp, kine = current_temperature( mol, self.ndeg )
        mol.get_grad()
        self.cacc = - mol.grad / mol.mass * 100.0
        if( self.rcom ):
            self.cacc -= numpy.sum( self.cacc * self.proj, axis = 0 ) * self.proj
        self.time = 0.0
        self.istp = 0
        self.log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( self.time, mol.func, kine, mol.func + kine, temp ) )


    def integrate( self, mol ):
        self.istp += 1
        self.time += self.step_size
        mol.coor += self.fr1 * mol.velo + self.fr2 * self.cacc
        r1 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) ) * mol.actv.astype( numpy.float64 )
        r2 = numpy.random.normal( 0.0, 1.0, ( mol.natm, 3 ) ) * mol.actv.astype( numpy.float64 )
        mol.coor += self.sdev * self.sr * r1
        oacc = self.c0 * mol.velo + self.fv1 * self.cacc + self.sdev * self.sv * ( self.cv1 * r1 + self.cv2 * r2 )
        mol.get_grad()
        self.cacc = - mol.grad / mol.mass * 100.0
        if( self.rcom ):
            self.cacc -= numpy.sum( self.cacc * self.proj, axis = 0 ) * self.proj
        mol.velo = oacc + self.fv2 * self.cacc
#        if( self.rcom ):
#            mol.velo -= numpy.sum( mol.velo * self.proj, axis = 0 ) * self.proj
        temp, kine = current_temperature( mol, self.ndeg )
        if( self.istp % self.print_frequency == 0 ):
            self.log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf\n"%( self.time, mol.func, kine, mol.func + kine, temp ) )
