import  sys
import  math
import  numpy
import  typing
import  ctypes
import  os


cwd = os.path.abspath( os.path.dirname( __file__ ) ) + os.sep


def steepest_descent( mol: object,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 15.,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    fdsc.write( "---------------------------------------- Minimization (SD)\n\n" )
    ndf = 3 * mol.actv.sum()
    fdsc.write( "Degrees of Freedom: %20ld\n"%( ndf ) )
    fdsc.write( "Step Number:        %20d\n"%( step_number ) )
    fdsc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdsc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    ndf = math.sqrt( ndf )
    mol.get_grad()
    norm = numpy.linalg.norm( mol.grad )
    if( norm > step_size ):
        ssiz = step_size
    elif( norm > gradient_tolerance ):
        ssiz = norm
    else:
        ssiz = gradient_tolerance
    grms = norm / ndf
    fdsc.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    fdsc.write( "-" * 70 + "\n" )
    fdsc.write( "%30.5lf%20.8lf%20.10lf\n"%( mol.func, grms, ssiz ) )
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
        grms = norm / ndf
        itr += 1
        if( itr % print_frequency == 0 ):
            fdsc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr, mol.func, grms, ssiz ) )
        mol.current_step( itr )
    if( itr % print_frequency != 0 ):
        fdsc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr + 1, mol.func, grms, ssiz ) )
    fdsc.write( "-" * 70 + "\n\n" )

# =================================================================================================

def fire( mol: object,
        step_number: typing.Optional[int] = 1000,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        gradient_tolerance: typing.Optional[float] = 1.5,
        mixing_alpha: typing.Optional[float] = 0.1,
        delay_step: typing.Optional[int] = 5,
        exit_uphill: typing.Optional[bool] = False,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    fdsc.write( "---------------------------------------- Minimization (FIRE)\n\n" )
    ndeg = 3 * mol.actv.sum()
    fdsc.write( "Degrees of Freedom: %20ld\n"%( ndeg ) )
    fdsc.write( "Step Number:        %20d\n"%( step_number ) )
    fdsc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdsc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    fdsc.write( "Checking UpHill:    %20s\n"%( exit_uphill ) )
    fdsc.write( "Mixing Alpha:       %20.10lg\n"%( mixing_alpha ) )
    fdsc.write( "Delay Step:         %20d\n\n"%( delay_step ) )
    fdsc.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Displacement" ) )
    fdsc.write( "-" * 70 + "\n" )
    ndeg = math.sqrt( ndeg )
    nstp = 0
    ssiz = step_size
    alph = mixing_alpha
    velo = numpy.zeros( ( mol.natm, 3 ), dtype=numpy.float64 )
    mol.get_grad()
    qfun = True
    norm = numpy.linalg.norm( mol.grad )
    grms = norm / ndeg
    fdsc.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
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
        norm = numpy.linalg.norm( mol.grad )
        grms = norm / ndeg
        if( exit_uphill ):
            if( lfun < mol.func ):
                fdsc.write( ">> search become uphill!\n" )
                qfun = False
                mol.coor -= step

        itr += 1
        if( itr % print_frequency == 0 ):
            fdsc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr, mol.func, grms, ssiz ) )
        mol.current_step( itr )
    if( itr % print_frequency != 0 ):
        fdsc.write( "%10d%20.5lf%20.10lf%20.10lf\n"%( itr + 1, mol.func, grms, ssiz ) )
    fdsc.write( "-" * 70 + "\n\n" )

# =================================================================================================

def cgplus( mol: object,
        step_number: typing.Optional[int] = 1000,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        method: typing.Optional[str] = "Polak-Ribiere", 
        restart: typing.Optional[bool] = True,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    global  cwd
    nsel = mol.actv.sum()
    size = 3 * nsel
    fdsc.write( "------------------------------------------ Minimization (CG+)\n\n" )
    fdsc.write( "Degrees of Freedom:   %20ld\n"%( size ) )
    fdsc.write( "Step Number:          %20d\n"%( step_number ) )
    fdsc.write( "Print Frequency:      %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance:   %20.10lg\n"%( gradient_tolerance ) )
    fdsc.write( "Method:             %22s\n\n"%( method ) )
    fdsc.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    fdsc.write( "-" * 50 + "\n" )
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
    ndeg = math.sqrt( size )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    mol.get_grad()
    grms = numpy.linalg.norm( mol.grad ) / ndeg
    coor = mol.coor[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    grad = mol.grad[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    dire = ( ctypes.c_double * size )()
    gold = ( ctypes.c_double * size )()
    work = ( ctypes.c_double * size )()
    iflg = ( ctypes.c_int )()
    fdsc.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
    itr  = 0
    while( itr < step_number and grms > gradient_tolerance ):
        dlib.cgp_cgfam_( ctypes.c_int( size ), coor, ctypes.c_double( mol.func ), grad, dire, gold,
                ctypes.c_double( gradient_tolerance ), work, iflg, ctypes.c_int( rest ), ctypes.c_int( meth ) )
        if( iflg == -3 ):
            fdsc.write( "\n -- Improper input parameters...\n" )
            itr = step_number + 1
        elif( iflg == -2 ):
            fdsc.write( "\n -- Descent was not obtained...\n" )
            itr = step_number + 1
        elif( iflg == -1 ):
            fdsc.write( "\n -- Line Search failure...\n" )
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
            fdsc.write( "%10d%20.5lf%20.10lf\n"%( itr, mol.func, grms ) )
        mol.current_step( itr )
    if( itr % print_frequency != 0 ):
        fdsc.write( "%10d%20.5lf%20.10lf\n"%( itr + 1, mol.func, grms ) )
    fdsc.write( "-" * 50 + "\n\n" )

# =================================================================================================

def baker( mol: object, get_hess: typing.Callable,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        follow_mode: typing.Optional[int] = -1,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    """
    import  qm3.utils.hessian

    def get_hess( mol: object, step: int ):
        hes = qm3.utils.hessian.numerical( mol )
        mol.get_grad()
        return( hes )
    """
    actv = mol.actv.sum()
    size = 3 * actv
    if( follow_mode >= size or follow_mode < -1 ):
        follow_mode = -1
    fdsc.write( "---------------------------------------- Minimization (Baker)\n\n" )
    fdsc.write( "Degrees of Freedom: %20ld\n"%( size ) )
    fdsc.write( "Following Mode:     %20d\n"%( follow_mode ) )
    fdsc.write( "Step Number:        %20d\n"%( step_number ) )
    fdsc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdsc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    if( follow_mode > -1 ):
        fdsc.write( "%10s%20s%20s%20s\n"%( "Step", "Function", "Gradient", "Nneg,Fmode,Eval" ) )
        fdsc.write( "-" * 70 + "\n" )
    else:
        fdsc.write( "%10s%20s%20s%5s\n"%( "Step", "Function", "Gradient", "Nneg" ) )
        fdsc.write( "-" * 55 + "\n" )
    mstp = 1.0e-1
    lrge = 1.0e+6
    step = 50.0
    tol1 = 1.0e-4
    tol2 = 1.0e-8
    emax = 1.0e5
    emin = 1.0e-3
    mxit = 999
    ndeg = math.sqrt( size )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
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
            fdsc.write( "\n -- Too much lambda iterations...\n" )
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
            fdsc.write( "\n -- The step size is *very* small...\n" )
            flg = False
        if( tmp > step_size ):
            crd *= step_size / tmp

        itr += 1
        grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( itr % print_frequency == 0 ):
            if( follow_mode < 0 ):
                fdsc.write( "%10ld%20.5lf%20.10lf%5ld%10.2le\n"%( itr, mol.func, grms, nneg, tmp ) )
            else:
                fdsc.write( "%10ld%20.5lf%20.10lf%5ld%5ld%10.2lf%10.2le\n"%( itr, mol.func, grms, nneg, follow_mode, who, tmp ) )
        mol.current_step( itr )

    if( itr % print_frequency != 0 ):
        if( follow_mode < 0 ):
            fdsc.write( "%10ld%20.5lf%20.10lf%5ld%10.2le\n"%( itr, mol.func, grms, nneg, tmp ) )
        else:
            fdsc.write( "%10ld%20.5lf%20.10lf%5ld%5ld%10.2lf%10.2le\n"%( itr, mol.func, grms, nneg, follow_mode, who, tmp ) )
    if( follow_mode > -1 ):
        fdsc.write( "-" * 70 + "\n" )
    else:
        fdsc.write( "-" * 55 + "\n" )

# =================================================================================================

def rfo( mol: object, get_hess: typing.Callable,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        follow_mode: typing.Optional[int] = -1,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    """
    import  qm3.utils.hessian

    def get_hess( mol: object, step: int ):
        hes = qm3.utils.hessian.numerical( mol )
        mol.get_grad()
        return( hes )
    """
    actv = mol.actv.sum()
    size = 3 * actv
    if( follow_mode >= size or follow_mode < -1 ):
        follow_mode = -1
    fdsc.write( "---------------------------------------- Minimization (RFO)\n\n" )
    fdsc.write( "Degrees of Freedom: %20ld\n"%( size ) )
    fdsc.write( "Following Mode:     %20d\n"%( follow_mode ) )
    fdsc.write( "Step Number:        %20d\n"%( step_number ) )
    fdsc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdsc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance: %20.10lg\n\n"%( gradient_tolerance ) )
    fdsc.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    fdsc.write( "-" * 50 + "\n" )
    tol2 = 1.0e-8
    ndeg = math.sqrt( size )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
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
        grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( itr % print_frequency == 0 ):
                fdsc.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, mol.func, grms, tmp ) )
        mol.current_step( itr )
    if( itr % print_frequency != 0 ):
        fdsc.write( "%10ld%20.5lf%20.10lf%10.2le\n"%( itr, mol.func, grms, tmp ) )
    fdsc.write( "-" * 50 + "\n" )

