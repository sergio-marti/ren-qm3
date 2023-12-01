import  math
import  numpy
import  typing
import  pickle
import  sys


def differential_evolution( function: typing.Callable,
        boundaries: list,
        step_number: typing.Optional[int] = 1000,
        step_tolerance: typing.Optional[float] = 1.0e-6,
        population_size: typing.Optional[int] = 20,
        mutation_factor: typing.Optional[float] = 0.8,
        crossover_probability: typing.Optional[float] = 0.75,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        checkpointing: typing.Optional[bool] = False ) -> tuple:
    # -------------------------------------------------------------------------
    dimension = boundaries[0].shape[0]
    population_size = max( population_size, dimension * 2 )
    mutation_factor = min( max( mutation_factor, 0.1 ), 1.0 )
    crossover_probability = min( max( crossover_probability, 0.1 ), 1.0 )
    # -------------------------------------------------------------------------
    log_file.write( "---------------------------------------- Genetic Minimization (DE: rand/1+bin)\n\n" )
    log_file.write( "Degrees of Freedom: %20ld\n"%( dimension ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Tolerance:     %20.10lg\n"%( step_tolerance ) )
    log_file.write( "Population:         %20d\n"%( population_size ) )
    log_file.write( "Mutation Factor:    %20.10lg\n"%( mutation_factor ) )
    log_file.write( "Crossover Prob.:    %20.10lg\n\n"%( crossover_probability ) )
    log_file.write( "%10s%30s\n"%( "Step", "Function" ) )
    log_file.write( "-" * 40 + "\n" )
    numpy.random.seed()
    minc = boundaries[0].copy()
    disp = numpy.abs( boundaries[1] - boundaries[0] )
    coor = numpy.random.random( ( population_size, dimension ) )
    func = numpy.zeros( population_size, dtype=numpy.float64 )
    for i in range( population_size ):
        func[i] = function( minc + disp * coor[i] )
    ok_fun = numpy.min( func )
    ok_crd = coor[numpy.argmin( func )]
    ok_stp = 2.0 * step_tolerance
    log_file.write( "%10s%30.10lf\n"%( "", ok_fun ) )
    it = 0
    ff = ok_fun
    while( it < step_number and ok_stp > step_tolerance ):
        for i in range( population_size ):
            # -------------------------------------------------------------------------
            # rand/1 + binomial
            a, b, c = numpy.random.choice( [ j for j in range( population_size ) if j != i ], 3, replace = False )
            trial = numpy.zeros( dimension, dtype=numpy.float64 )
            for j in range( dimension ):
                if( numpy.random.random() < crossover_probability ):
                    trial[j] = min( max( coor[a,j] + mutation_factor * ( coor[b,j] - coor[c,j] ), 0.0 ), 1.0 )
                else:
                    trial[j] = coor[i,j]
            # -------------------------------------------------------------------------
            cfun = function( minc + disp * trial )
            if( cfun < func[i] ):
                func[i] = cfun
                coor[i,:] = trial.copy()
                if( cfun < ok_fun ):
                    ok_stp = numpy.linalg.norm( ok_crd - trial ) / float( dimension )
                    ok_fun = cfun
                    ok_crd = trial.copy()
        if( ff > ok_fun ):
            ff = ok_fun
            log_file.write( "%10d%30.10lf"%( it, ok_fun ) + " (%.1le)"%( ok_stp ) + "\n" )
            if( checkpointing ):
                fd = open( "diffevo.chk", "wb" )
                pickle.dump( minc + disp * ok_crd, fd )
                fd.close()
        it += 1
    log_file.write( "-" * 40 + "\n" )
    return( ok_fun, minc + disp * ok_crd )


def fire( function: typing.Callable,
        initial_guess: numpy.array,
        step_number: typing.Optional[int] = 1000,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 0.001,
        mixing_alpha: typing.Optional[float] = 0.1,
        delay_step: typing.Optional[int] = 5,
        exit_uphill: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout ) -> tuple:
    """
    Phys. Rev. Lett. v97, p170201 (2006) [doi:10.1103/PhysRevLett.97.170201]
    """
    dimension = initial_guess.shape[0]
    log_file.write( "---------------------------------------- Minimization (FIRE)\n\n" )
    log_file.write( "Degrees of Freedom: %20ld\n"%( dimension ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    log_file.write( "Checking UpHill:    %20s\n"%( exit_uphill ) )
    log_file.write( "Mixing Alpha:       %20.10lg\n"%( mixing_alpha ) )
    log_file.write( "Delay Step:         %20d\n\n"%( delay_step ) )
    log_file.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    log_file.write( "-" * 70 + "\n" )
    nstp = 0
    ssiz = step_size
    alph = mixing_alpha
    velo = numpy.zeros( dimension, dtype=numpy.float64 )
    coor = initial_guess.copy()
    func, grad = function( coor )
    qfun = True
    norm = numpy.linalg.norm( grad )
    log_file.write( "%30.5lf%20.10lf\n"%( func, norm ) )
    itr  = 0
    while( itr < step_number and norm > gradient_tolerance and qfun ):
        if( - numpy.sum( velo * grad ) > 0.0 ):
            vsiz = numpy.linalg.norm( velo )
            velo = ( 1.0 - alph ) * velo - alph * grad / norm * vsiz
            if( nstp > delay_step ):
                ssiz = min( ssiz * 1.1, step_size )
                alph *= 0.99
            nstp += 1
        else:
            alph = mixing_alpha
            ssiz *= 0.5
            nstp = 0
            velo = numpy.zeros( dimension, dtype=numpy.float64 )
        velo -= ssiz * grad
        step = ssiz * velo
        tmp  = numpy.linalg.norm( step )
        if( tmp > ssiz ):
            step *= ssiz / tmp
        coor += step

        lfun = func
        func, grad = function( coor )
        norm = numpy.linalg.norm( grad )
        if( exit_uphill ):
            if( lfun < func ):
                log_file.write( ">> search become uphill!\n" )
                qfun = False
                coor -= step

        itr += 1
        if( itr % print_frequency == 0 ):
            log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr, func, norm, ssiz ) )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr + 1, func, norm, ssiz ) )
    log_file.write( "-" * 70 + "\n\n" )
    return( func, coor )


def rfo( function: typing.Callable,
        initial_guess: numpy.array,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 0.001,
        follow_mode: typing.Optional[int] = -1,
        log_file: typing.Optional[typing.IO] = sys.stdout ) -> tuple:
    """
    J. Phys. chem. v89, p52 (1985) [doi:10.1021/j100247a015]

    """
    dimension = initial_guess.shape[0]
    if( follow_mode >= dimension or follow_mode < -1 ):
        follow_mode = -1
    log_file.write( "---------------------------------------- Minimization (RFO)\n\n" )
    log_file.write( "Degrees of Freedom: %20ld\n"%( dimension ) )
    log_file.write( "Following Mode:     %20d\n"%( follow_mode ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    log_file.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    log_file.write( "-" * 50 + "\n" )
    coor = initial_guess.copy()
    tol2 = 1.0e-8
    grms = gradient_tolerance * 2.0
    crd  = numpy.zeros( dimension, dtype=numpy.float64 )
    new  = 0.5 * numpy.ones( dimension, dtype=numpy.float64 )
    if( follow_mode > -1 ):
        new[follow_mode] *= -1.0
    flg  = True
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance and flg ):
        coor -= crd
        func, grad, hess = function( coor )
        val, vec = numpy.linalg.eigh( hess )
        idx = numpy.argsort( val )
        val = val[idx]
        vec = vec[:,idx]
        grd = numpy.dot( vec.T, grad )
        val = new * ( numpy.fabs( val ) + numpy.sqrt( val * val + 4.0 * grd * grd ) )
        crd = numpy.dot( vec, grd * ( 1.0 / val ) )
        tmp = numpy.linalg.norm( crd )
        if( tmp > step_size ):
            crd *= step_size / tmp
        itr += 1
        grms = numpy.linalg.norm( grad )
        if( itr % print_frequency == 0 or itr == 1 ):
                log_file.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, fun, grms, tmp ) )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, fun, grms, tmp ) )
    log_file.write( "-" * 50 + "\n" )
    return( func, coor )



if( __name__ == "__main__" ):

    def func_muller_brown( crd ):
        A  = [ -200.0, -100.0, -170.0, 15.0 ]
        a  = [ -1.0, -1.0, -6.5, 0.7 ]
        b  = [ 0.0, 0.0, 11.0, 0.6 ]
        c  = [ -10.0, -10.0, -6.5, 0.7 ]
        xo = [ 1.0, 0.0, -0.5, -1.0 ]
        yo = [ 0.0, 0.5, 1.5, 1.0 ]
        ff = 0.0
        for i in [0, 1, 2, 3]:
            ff += A[i] * math.exp( a[i] * math.pow( crd[0] - xo[i], 2.0 ) + b[i] * ( crd[0] - xo[i] ) * ( crd[1] - yo[i] ) + c[i] * math.pow( crd[1] - yo[i], 2.0 ) )
        return( ff )

    def grad_muller_brown( crd ):
        A  = [ -200.0, -100.0, -170.0, 15.0 ]
        a  = [ -1.0, -1.0, -6.5, 0.7 ]
        b  = [ 0.0, 0.0, 11.0, 0.6 ]
        c  = [ -10.0, -10.0, -6.5, 0.7 ]
        xo = [ 1.0, 0.0, -0.5, -1.0 ]
        yo = [ 0.0, 0.5, 1.5, 1.0 ]
        ff = 0.0
        gg = numpy.array( [ .0, .0 ] )
        for i in [0, 1, 2, 3]:
            f = A[i] * math.exp( a[i] * math.pow( crd[0] - xo[i], 2.0 ) + b[i] * ( crd[0] - xo[i] ) * ( crd[1] - yo[i] ) + c[i] * math.pow( crd[1] - yo[i], 2.0 ) )
            ff += f
            gg[0] += f * ( 2.0 * a[i] * ( crd[0] - xo[i] ) + b[i] * ( crd[1] - yo[i] ) )
            gg[1] += f * ( b[i] * ( crd[0] - xo[i] ) + 2.0 * c[i] * ( crd[1] - yo[i] ) )
        return( ff, gg )

    def hess_muller_brown( crd ):
        A  = [ -200.0, -100.0, -170.0, 15.0 ]
        a  = [ -1.0, -1.0, -6.5, 0.7 ]
        b  = [ 0.0, 0.0, 11.0, 0.6 ]
        c  = [ -10.0, -10.0, -6.5, 0.7 ]
        xo = [ 1.0, 0.0, -0.5, -1.0 ]
        yo = [ 0.0, 0.5, 1.5, 1.0 ]
        ff = 0.0
        gg = numpy.array( [ .0, .0 ] )
        hh = numpy.array( [ [ .0, .0 ], [ .0, .0 ] ] )
        for i in [0, 1, 2, 3]:
            f = A[i] * math.exp( a[i] * math.pow( crd[0] - xo[i], 2.0 ) + b[i] * ( crd[0] - xo[i] ) * ( crd[1] - yo[i] ) + c[i] * math.pow( crd[1] - yo[i], 2.0 ) )
            ff += f
            gg[0] += f * ( 2.0 * a[i] * ( crd[0] - xo[i] ) + b[i] * ( crd[1] - yo[i] ) )
            gg[1] += f * ( b[i] * ( crd[0] - xo[i] ) + 2.0 * c[i] * ( crd[1] - yo[i] ) )
            hh[0,0] += f * ( 2.0 * a[i] + math.pow( 2.0 * a[i] * ( crd[0] - xo[i] ) + b[i] * ( crd[1] - yo[i] ), 2.0 ) )
            hh[1,1] += f * ( 2.0 * c[i] + math.pow( 2.0 * c[i] * ( crd[1] - yo[i] ) + b[i] * ( crd[0] - xo[i] ), 2.0 ) )
            t = f * ( b[i] + ( 2.0 * a[i] * ( crd[0] - xo[i] ) + b[i] *( crd[1] - yo[i] ) ) * ( b[i] * ( crd[0] - xo[i] ) + 2.0 * c[i] * ( crd[1] - yo[i] ) ) )
            hh[0,1] += t
            hh[1,0] += t
        return( ff, gg, hh )



    fun, crd = differential_evolution( func_muller_brown, [ numpy.array( [ -0.75, 1.2 ] ), numpy.array( [ -0.25, 1.6 ] ) ] )
    print( crd, fun, numpy.linalg.norm( grad_muller_brown( crd )[1] ) )

    fun, crd = fire( grad_muller_brown, numpy.array( [ -0.75, 1.2 ] ) )
    print( crd, fun, numpy.linalg.norm( grad_muller_brown( crd )[1] ) )

    fun, crd = rfo( hess_muller_brown, numpy.array( [ -0.75, 1.2 ] ) )
    print( crd, fun, numpy.linalg.norm( grad_muller_brown( crd )[1] ) )

    import  scipy.optimize
    print( 80 * "=" )
    print( scipy.optimize.differential_evolution( func_muller_brown, [ ( -0.75, -0.25 ), ( 1.2, 1.6 ) ],
                strategy = "rand1bin" ) )
    print( 80 * "=" )
    print( scipy.optimize.minimize( func_muller_brown, [ -0.75, 1.2 ], method = "BFGS",
                jac = lambda x: grad_muller_brown( x )[1] ) )
    print( 80 * "=" )
    print( scipy.optimize.minimize( func_muller_brown, [ -0.75, 1.2 ], method = "Newton-CG",
                jac  = lambda x: grad_muller_brown( x )[1],
                hess = lambda x: hess_muller_brown( x )[2] ) )


