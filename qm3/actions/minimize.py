import  sys
import  math
import  numpy
import  typing
import  ctypes
import  os


# Gaussian tolerances by default in optimizations:
#
# 0.000450 for Maximum Force (2.23 kJ/mol.A)
# 0.000300 for RMS     Force (1.49 kJ/mol.A)
# -----------------------------------------------


cwd = os.path.abspath( os.path.dirname( __file__ ) ) + os.sep


def fake_cs( self: object, step: int ):
    sys.stdout.flush()


def steepest_descent( mol: object,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 15.,
        use_maxgrad: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    log_file.write( "---------------------------------------- Minimization (SD)\n\n" )
    ndf = 3 * mol.actv.sum()
    log_file.write( "Degrees of Freedom: %20ld\n"%( ndf ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Use Maxgradient:    %20s\n"%( use_maxgrad ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    if( use_maxgrad ):
        ndf = math.sqrt( 3.0 )
    else:
        ndf = math.sqrt( ndf )
    mol.get_grad()
    norm = numpy.linalg.norm( mol.grad )
    if( norm > step_size ):
        ssiz = step_size
    elif( norm > gradient_tolerance ):
        ssiz = norm
    else:
        ssiz = gradient_tolerance
    if( use_maxgrad ):
        grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) ) / ndf
    else:
        grms = norm / ndf
    log_file.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    log_file.write( "-" * 70 + "\n" )
    log_file.write( "%30.5lf%20.8lf%20.10lf\n"%( mol.func, grms, ssiz ) )
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance ):
        mol.coor -= mol.grad / norm * ssiz
        mol.get_grad()
        norm = numpy.linalg.norm( mol.grad )
        if( norm > step_size ):
            ssiz = step_size
        elif( norm > gradient_tolerance ):
            ssiz = norm
        else:
            ssiz = gradient_tolerance
        if( use_maxgrad ):
            grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) ) / ndf
        else:
            grms = norm / ndf
        itr += 1
        if( itr % print_frequency == 0 ):
            log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr, mol.func, grms, ssiz ) )
        current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr + 1, mol.func, grms, ssiz ) )
    log_file.write( "-" * 70 + "\n\n" )

# =================================================================================================

def fire( mol: object,
        step_number: typing.Optional[int] = 1000,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        mixing_alpha: typing.Optional[float] = 0.1,
        delay_step: typing.Optional[int] = 5,
        exit_uphill: typing.Optional[bool] = False,
        use_maxgrad: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    Phys. Rev. Lett. v97, p170201 (2006) [doi:10.1103/PhysRevLett.97.170201]
    """
    log_file.write( "---------------------------------------- Minimization (FIRE)\n\n" )
    ndeg = 3 * mol.actv.sum()
    log_file.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Use Maxgradient:    %20s\n"%( use_maxgrad ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    log_file.write( "Checking UpHill:    %20s\n"%( exit_uphill ) )
    log_file.write( "Mixing Alpha:       %20.10lg\n"%( mixing_alpha ) )
    log_file.write( "Delay Step:         %20d\n\n"%( delay_step ) )
    log_file.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    log_file.write( "-" * 70 + "\n" )
    if( use_maxgrad ):
        ndeg = math.sqrt( 3.0 )
        if( gradient_tolerance == 1.5 ):
            gradient_tolerance = 2.2
            log_file.write( ">> switching gradient_tolerance to 2.2\n" )
    else:
        ndeg = math.sqrt( ndeg )
    nstp = 0
    ssiz = step_size
    alph = mixing_alpha
    velo = numpy.zeros( ( mol.natm, 3 ), dtype=numpy.float64 )
    mol.get_grad()
    qfun = True
    if( use_maxgrad ):
        norm = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) )
    else:
        norm = numpy.linalg.norm( mol.grad )
    grms = norm / ndeg
    log_file.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance and qfun ):
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
            velo = numpy.zeros( ( mol.natm, 3 ), dtype=numpy.float64 )
        velo -= ssiz * mol.grad
        step = ssiz * velo
        tmp  = numpy.linalg.norm( step )
        if( tmp > ssiz ):
            step *= ssiz / tmp
        mol.coor += step

        lfun = mol.func
        mol.get_grad()
        if( use_maxgrad ):
            norm = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) )
        else:
            norm = numpy.linalg.norm( mol.grad )
        grms = norm / ndeg
        if( exit_uphill ):
            if( lfun < mol.func ):
                log_file.write( ">> search become uphill!\n" )
                qfun = False
                mol.coor -= step

        itr += 1
        if( itr % print_frequency == 0 ):
            log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr, mol.func, grms, ssiz ) )
        current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr + 1, mol.func, grms, ssiz ) )
    log_file.write( "-" * 70 + "\n\n" )

# =================================================================================================

def cgplus( mol: object,
        step_number: typing.Optional[int] = 1000,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        method: typing.Optional[str] = "Polak-Ribiere", 
        restart: typing.Optional[bool] = True,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    global  cwd
    size = 3 * mol.actv.sum()
    log_file.write( "------------------------------------------ Minimization (CG+)\n\n" )
    log_file.write( "Degrees of Freedom:   %20ld\n"%( size ) )
    log_file.write( "Step Number:          %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:      %20d\n"%( print_frequency ) )
    log_file.write( "Gradient Tolerance:   %20.10lg\n"%( gradient_tolerance ) )
    log_file.write( "Method:             %22s\n\n"%( method ) )
    log_file.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    log_file.write( "-" * 50 + "\n" )
    ndeg = math.sqrt( size )
    rest = int( restart )
    meth = 2
    kind = { "Fletcher-Reeves" : 1, "Polak-Ribiere" : 2, "Positive Polak-Ribiere": 3 }
    if( method in kind ):
        meth = kind[method]
    dlib = ctypes.CDLL( cwd + "_cgplus.so" )
    dlib.cgp_cgfam_.argtypes = [ 
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_int ) ]
    dlib.cgp_cgfam_.restype = None
    sele = numpy.flatnonzero( mol.actv.ravel() )
    mol.get_grad()
    grms = numpy.linalg.norm( mol.grad ) / ndeg
    coor = mol.coor[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    grad = mol.grad[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    dire = ( ctypes.c_double * size )()
    gold = ( ctypes.c_double * size )()
    work = ( ctypes.c_double * size )()
    iflg = ( ctypes.c_int )()
    log_file.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance ):
        dlib.cgp_cgfam_( ctypes.c_int( size ), coor, ctypes.c_double( mol.func ), grad, dire, gold,
                ctypes.c_double( gradient_tolerance ), work, iflg, ctypes.c_int( rest ), ctypes.c_int( meth ) )
        if( iflg == -3 ):
            log_file.write( "\n -- Improper input parameters...\n" )
            itr = step_number + 1
        elif( iflg == -2 ):
            log_file.write( "\n -- Descent was not obtained...\n" )
            itr = step_number + 1
        elif( iflg == -1 ):
            log_file.write( "\n -- Line Search failure...\n" )
            itr = step_number + 1
        else:
            while( iflg == 2 ):
                dlib.cgp_cgfam_( ctypes.c_int( size ), coor, ctypes.c_double( mol.func ), grad, dire, gold,
                        ctypes.c_double( gradient_tolerance ), work, iflg, ctypes.c_int( rest ), ctypes.c_int( meth ) )
            l = 0
            for i in sele:
                for j in [0, 1, 2]:
                    mol.coor[i,j] = coor[l]
                    l += 1
            mol.get_grad()
            grms = numpy.linalg.norm( mol.grad ) / ndeg
            grad = mol.grad[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
        itr += 1
        if( itr % print_frequency == 0 ):
            log_file.write( "%10d%20.5lf%20.10lf\n"%( itr, mol.func, grms ) )
        current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf\n"%( itr + 1, mol.func, grms ) )
    log_file.write( "-" * 50 + "\n\n" )

# =================================================================================================

def lbfgs( mol: object,
        step_number: typing.Optional[int] = 1000,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    global  cwd
    size = 3 * mol.actv.sum()
    log_file.write( "------------------------------------------ Minimization (L-BFGS: Fortran)\n\n" )
    log_file.write( "Degrees of Freedom:   %20ld\n"%( size ) )
    log_file.write( "Step Number:          %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:      %20d\n"%( print_frequency ) )
    log_file.write( "Use Maxgradient:      %20s\n"%( use_maxgrad ) )
    log_file.write( "Gradient Tolerance:   %20.10lg\n\n"%( gradient_tolerance ) )
    log_file.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    log_file.write( "-" * 50 + "\n" )
    dlib = ctypes.CDLL( cwd + "_lbfgsb.so" )
    dlib.lbfgsb_setulb_.argtypes = [ 
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_double ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_char ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_char ),
        ctypes.POINTER( ctypes.c_bool ),
        ctypes.POINTER( ctypes.c_int ),
        ctypes.POINTER( ctypes.c_double ) ]
    dlib.lbfgsb_setulb_.restype = None
    nmx = size
    mmx = 9
    crd = ( ctypes.c_double * nmx )()
    low = ( ctypes.c_double * nmx )()
    upp = ( ctypes.c_double * nmx )()
    nbd = ( ctypes.c_int * nmx )()
    grd = ( ctypes.c_double * nmx )()
    wrk = ( ctypes.c_double * ( ( 2 * mmx + 4 ) * nmx + 12 * mmx * ( 1 + mmx ) ) )()
    iwa = ( ctypes.c_int * ( 3 * nmx ) )()
    tsk = ( ctypes.c_char * 60 )()
    tsk.raw = b"START"
    csv = ( ctypes.c_char * 60 )()
    lsv = ( ctypes.c_bool * 4 )()
    isv = ( ctypes.c_int * 44 )()
    dsv = ( ctypes.c_double * 29 )()
    ndeg = math.sqrt( size )
    sele = numpy.flatnonzero( mol.actv.ravel() )
    mol.get_grad()
    grms = numpy.linalg.norm( mol.grad ) / ndeg
    # NO BOUNDARIES -----------------------------------
    k = 0
    for i in sele:
        for j in [0, 1, 2]:
            low[k] = 0.0
            upp[k] = 0.0
            nbd[k] = 0
            crd[k] = mol.coor[i,j]
            grd[k] = mol.grad[i,j]
            k += 1
    # -------------------------------------------------
    log_file.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
    itr = 0
    while( itr < step_number and grms > gradient_tolerance ):
        dlib.lbfgsb_setulb_( ctypes.c_int( nmx ), ctypes.c_int( mmx ),
            crd, low, upp, nbd, ctypes.c_double( mol.func ), grd,
            ctypes.c_double( 1.e3 ), ctypes.c_double( gradient_tolerance ),
            wrk, iwa, tsk, ctypes.c_int( -1 ), csv, lsv, isv, dsv )
        if( tsk.raw[0:4] in [ b"CONV", b"STOP", b"ERRO", b"ABNO" ] ):
            it = step_number + 1
        elif( tsk.raw[0:2] == b"FG" ):
            k = 0
            for i in sele:
                for j in [0, 1, 2]:
                    mol.coor[i,j] = crd[k]
                    k += 1
            mol.get_grad()
            k = 0
            for i in sele:
                for j in [0, 1, 2]:
                    grd[k] = mol.grad[i,j]
                    k += 1
            grms = numpy.linalg.norm( mol.grad ) / ndeg
            itr += 1
            if( itr % print_frequency == 0 ):
                log_file.write( "%10d%20.5lf%20.10lf\n"%( itr, mol.func, grms ) )
            current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf\n"%( itr + 1, mol.func, grms ) )
    log_file.write( "-" * 50 + "\n\n" )


def l_bfgs( mol: object, 
        step_number: typing.Optional[int] = 1000,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        use_maxgrad: typing.Optional[bool] = False,
        history: typing.Optional[int] = 9,
        exit_uphill: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    log_file.write( "---------------------------------------- Minimization (L-BFGS: Python)\n\n" )
    sele = numpy.flatnonzero( mol.actv.ravel() )
    size = 3 * sele.shape[0]
    log_file.write( "Degrees of Freedom: %20ld\n"%( size ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    log_file.write( "Use Maxgradient:    %20s\n"%( use_maxgrad ) )
    log_file.write( "Number of Updates:  %20d\n"%( history ) )
    log_file.write( "Checking UpHill:    %20s\n\n"%( exit_uphill ) )
    log_file.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    log_file.write( "-" * 50 + "\n" )
    if( use_maxgrad ):
        ndeg = math.sqrt( 3.0 )
        if( gradient_tolerance == 1.5 ):
            gradient_tolerance = 2.2
            log_file.write( ">> switching gradient_tolerance to 2.2\n" )
    else:
        ndeg = math.sqrt( size )
    aux   = numpy.zeros( history, dtype=numpy.float64 )
    rho   = numpy.zeros( history, dtype=numpy.float64 )
    dg    = numpy.zeros( ( history, size ), dtype=numpy.float64 )
    dx    = numpy.zeros( ( history, size ), dtype=numpy.float64 )
    hscal = 1.0
    mol.get_grad()
    if( use_maxgrad ):
        grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) ) / ndeg
    else:
        grms = numpy.linalg.norm( mol.grad ) / ndeg
    qfun = True
    log_file.write( "%10s%20.5lf%20.8lf\n"%( "", mol.func, grms ) )
    itr = 0
    while( itr < step_number and grms > gradient_tolerance and qfun ):
        if( itr > history ):
            dx  = numpy.roll(  dx, -1, axis = 0 )
            dg  = numpy.roll(  dg, -1, axis = 0 )
            rho = numpy.roll( rho, -1 )
        if( itr > 0 ):
            j      = min( itr, history ) - 1
            dx[j]  = mol.coor[sele].ravel() - ox
            dg[j]  = mol.grad[sele].ravel() - og
            hgx    = numpy.sum( dg[j] * dx[j] )
            hgg    = numpy.sum( dg[j] * dg[j] )
            rho[j] = 1.0 / hgx
            hscal  = hgx / hgg
        ox   = mol.coor[sele].ravel()
        og   = mol.grad[sele].ravel()
        step = - og
        if( itr > 0 ):
            for j in reversed( range( min( itr, history ) ) ):
                aux[j] = rho[j] * numpy.sum( step * dx[j] )
                step  -= aux[j] * dg[j]
            step *= hscal
            for j in range( min( itr, history ) ):
                aux[j] -= rho[j] * numpy.sum( step * dg[j] )
                step   += aux[j] * dx[j]
        tmp = numpy.linalg.norm( step )
        if( tmp > step_size ):
            step *= step_size / tmp
        mol.coor[sele] += step.reshape( ( sele.shape[0], 3 ) )
        lfun = mol.func
        mol.get_grad()
        if( use_maxgrad ):
            grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) ) / ndeg
        else:
            grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( exit_uphill ):
            if( lfun < mol.func ):
                log_file.write( ">> search become uphill!\n" )
                qfun = False
                mol.coor[sele] -= step.reshape( ( sele.shape[0], 3 ) )
        itr += 1
        if( itr % print_frequency == 0 ):
            log_file.write( "%10d%20.5lf%20.10lf\n"%( itr, mol.func, grms ) )
        current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10d%20.5lf%20.10lf\n"%( itr + 1, mol.func, grms ) )
    log_file.write( "-" * 50 + "\n" )

# =================================================================================================

def baker( mol: object,
        get_hess: typing.Callable,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        follow_mode: typing.Optional[int] = -1,
        use_maxgrad: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    baker_optimization.F90 fDynamo module
    J. Chem. Phys. v75, p2800 (1981) [doi:10.1063/1.442352]
    J. Phys. Chem. V87, p2745 (1983) [doi:10.1021/j100238a013]

    import  qm3.utils
    import  qm3.utils.hessian

    def calc_hess( self: object, step: int ):
        if( step % 10 == 0 ):
            self.hess = qm3.utils.hessian.numerical( self )
            qm3.utils.hessian.manage( self, self.hess )
            self.get_grad()
        else:
            self.get_grad()
            qm3.utils.hessian.manage( self, self.hess, should_update = True )
        return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) )
    """
    actv = mol.actv.sum()
    size = 3 * actv
    if( follow_mode >= size or follow_mode < -1 ):
        follow_mode = -1
    log_file.write( "---------------------------------------- Minimization (Baker)\n\n" )
    log_file.write( "Degrees of Freedom: %20ld\n"%( size ) )
    log_file.write( "Following Mode:     %20d\n"%( follow_mode ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Use Maxgradient:    %20s\n"%( use_maxgrad ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    if( follow_mode > -1 ):
        log_file.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Nneg,Fmode,Eval" ) )
        log_file.write( "-" * 70 + "\n" )
    else:
        log_file.write( "%10s%20s%20s%5s\n"%( "Step", "Function", "Gradient", "Nneg" ) )
        log_file.write( "-" * 55 + "\n" )
    mstp = 1.0e-1
    lrge = 1.0e+6
    step = 50.0
    tol1 = 1.0e-4
    tol2 = 1.0e-8
    emax = 1.0e5
    emin = 1.0e-3
    mxit = 999
    if( use_maxgrad ):
        ndeg = math.sqrt( 3.0 )
        if( gradient_tolerance == 1.5 ):
            gradient_tolerance = 2.2
            log_file.write( ">> switching gradient_tolerance to 2.2\n" )
    else:
        ndeg = math.sqrt( size )
    sele = numpy.flatnonzero( mol.actv.ravel() )
    grms = gradient_tolerance * 2.0
    crd  = numpy.zeros( ( actv, 3 ), dtype=numpy.float64 )
    flg  = True
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance and flg ):
        mol.coor[sele] += crd

        crd = numpy.zeros( ( actv, 3 ), dtype=numpy.float64 )
        hes = get_hess( mol, itr )
        val, vec = numpy.linalg.eigh( hes )
        idx = numpy.argsort( val )
        val = val[idx]
        vec = vec[:,idx]
        nneg = 0
        for i in range( size ):
            nneg += val[i] < 0.0
            if( math.fabs( val[i] ) < emin ):
                val[i] = numpy.sign( val[i] ) * emin
            elif( math.fabs( val[i] ) > emax ):
                val[i] = numpy.sign( val[i] ) * emax
        grd = mol.grad[sele].ravel().reshape( ( size, 1 ) )
        grd = numpy.dot( vec.T, grd )

        if( follow_mode > -1 ):
            who = val[follow_mode]
            if( math.fabs( grd[follow_mode] ) > tol1 ):
                lmbd = 0.5 * ( who + math.sqrt( who * who + 4.0 * grd[follow_mode] * grd[follow_mode] ) ) 
                lmbd = grd[follow_mode] / ( lmbd - val[follow_mode] )
            else:
                if( nneg == 1 ):
                    lmbd = - grd[follow_mode] / val[follow_mode]
                else:
                    lmbd = mstp
            crd = lmbd * vec[:,follow_mode]
            crd.shape = ( actv, 3 )

        if( follow_mode == 0 ):
            lowr = 1
        else:
            lowr = 0 
        lmbd = 0.0
        if( val[lowr] < 0.0 ):
            lmbd = val[lowr] - step
            l1   = val[lowr]
            l2   = - lrge
        tmp  = 0.0;
        for j in range( size ):
            if( j != follow_mode ):
                tmp += ( grd[j] * grd[j] ) / ( lmbd - val[j] )
        i = 0
        while( i < mxit and math.fabs( lmbd - tmp ) >= tol2 ):
            if( val[lowr] > 0.0 ):
                lmbd = tmp;
            else:
                if( tmp < lmbd ):
                    l1 = lmbd;
                if( tmp > lmbd ):
                    l2 = lmbd;
                if( l2 > - lrge ):
                    lmbd = 0.5 * ( l1 + l2 )
                elif( l2 == - lrge ):
                    lmbd -= step;
            tmp  = 0.0;
            for j in range( size ):
                if( j != follow_mode ):
                    tmp += ( grd[j] * grd[j] ) / ( lmbd - val[j] )
            i += 1
        if( i > mxit ):
            log_file.write( "\n -- Too much lambda iterations...\n" )
            flg = False

        if( follow_mode > -1 ):
            val[follow_mode] = lmbd - 1.0
            vec[:,follow_mode] = 0.0

        val.shape = ( size, 1 )
        tmp = numpy.dot( vec, grd / ( lmbd - val ) )
        tmp.shape = ( actv, 3 )
        crd += tmp
        tmp = numpy.linalg.norm( crd )
        if( tmp < tol2 ):
            log_file.write( "\n -- The step size is *very* small...\n" )
            flg = False
        if( tmp > step_size ):
            crd *= step_size / tmp

        itr += 1
        if( use_maxgrad ):
            grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) )
        else:
            grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( itr % print_frequency == 0 or itr == 1 ):
            if( follow_mode < 0 ):
                log_file.write( "%10ld%20.5lf%20.10lf%5ld%10.2le\n"%( itr, mol.func, grms, nneg, tmp ) )
            else:
                log_file.write( "%10ld%20.5lf%20.10lf%5ld%5ld%10.2lf%10.2le\n"%( itr, mol.func, grms, nneg, follow_mode, who, tmp ) )
        current_step( mol, itr )

    if( itr % print_frequency != 0 ):
        if( follow_mode < 0 ):
            log_file.write( "%10ld%20.5lf%20.10lf%5ld%10.2le\n"%( itr, mol.func, grms, nneg, tmp ) )
        else:
            log_file.write( "%10ld%20.5lf%20.10lf%5ld%5ld%10.2lf%10.2le\n"%( itr, mol.func, grms, nneg, follow_mode, who, tmp ) )
    if( follow_mode > -1 ):
        log_file.write( "-" * 70 + "\n" )
    else:
        log_file.write( "-" * 55 + "\n" )

# =================================================================================================

def rfo( mol: object,
        get_hess: typing.Callable,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        follow_mode: typing.Optional[int] = -1,
        use_maxgrad: typing.Optional[bool] = False,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = fake_cs ):
    """
    J. Phys. chem. v89, p52 (1985) [doi:10.1021/j100247a015]

    import  qm3.utils
    import  qm3.utils.hessian

    def calc_hess( self: object, step: int ):
        if( step % 10 == 0 ):
            self.hess = qm3.utils.hessian.numerical( self )
            qm3.utils.hessian.manage( self, self.hess )
            self.get_grad()
        else:
            self.get_grad()
            qm3.utils.hessian.manage( self, self.hess, should_update = True )
        return( return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) ) )
    """
    actv = mol.actv.sum()
    size = 3 * actv
    if( follow_mode >= size or follow_mode < -1 ):
        follow_mode = -1
    log_file.write( "---------------------------------------- Minimization (RFO)\n\n" )
    log_file.write( "Degrees of Freedom: %20ld\n"%( size ) )
    log_file.write( "Following Mode:     %20d\n"%( follow_mode ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Step Size:          %20.10lg\n"%( step_size ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "Use Maxgradient:    %20s\n"%( use_maxgrad ) )
    log_file.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    log_file.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    log_file.write( "-" * 50 + "\n" )
    tol2 = 1.0e-8
    if( use_maxgrad ):
        ndeg = math.sqrt( 3.0 )
        if( gradient_tolerance == 1.5 ):
            gradient_tolerance = 2.2
            log_file.write( ">> switching gradient_tolerance to 2.2\n" )
    else:
        ndeg = math.sqrt( size )
    sele = numpy.flatnonzero( mol.actv.ravel() )
    grms = gradient_tolerance * 2.0
    crd  = numpy.zeros( ( actv, 3 ), dtype=numpy.float64 )
    new  = 0.5 * numpy.ones( size, dtype=numpy.float64 )
    if( follow_mode > -1 ):
        new[follow_mode] *= -1.0
    flg  = True
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance and flg ):
        mol.coor[sele] -= crd
        hes = get_hess( mol, itr )
        val, vec = numpy.linalg.eigh( hes )
        idx = numpy.argsort( val )
        val = val[idx]
        vec = vec[:,idx]
        grd = mol.grad[sele].ravel()
        grd = numpy.dot( vec.T, grd )
        val = new * ( numpy.fabs( val ) + numpy.sqrt( val * val + 4.0 * grd * grd ) )
        crd = numpy.dot( vec, grd * ( 1.0 / val ) ).reshape( ( actv, 3 ) )
        tmp = numpy.linalg.norm( crd )
        if( tmp > step_size ):
            crd *= step_size / tmp
        itr += 1
        if( use_maxgrad ):
            grms = numpy.max( numpy.linalg.norm( mol.grad, axis = 1 ) ) / ndeg
        else:
            grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( itr % print_frequency == 0 or itr == 1 ):
                log_file.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, mol.func, grms, tmp ) )
        current_step( mol, itr )
    if( itr % print_frequency != 0 ):
        log_file.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, mol.func, grms, tmp ) )
    log_file.write( "-" * 50 + "\n" )

