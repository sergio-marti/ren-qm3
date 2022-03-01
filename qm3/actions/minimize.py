import  sys
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
    fdesc.write( "%30.5lf%20.8lf%20.10lf\n"%( mol.func, grms, ssiz ) )
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
    fdesc.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
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

def cgplus( mol: object,
        step_number: typing.Optional[int] = 1000,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        method: typing.Optional[str] = "Polak-Ribiere", 
        restart: typing.Optional[bool] = True,
        fdesc: typing.Optional = sys.stdout ):
    global  cwd
    nsel = mol.actv.sum()
    size = 3 * nsel
    fdesc.write( "------------------------------------------ Minimization (CG+)\n\n" )
    fdesc.write( "Degrees of Freedom:   %20ld\n"%( size ) )
    fdesc.write( "Step Number:          %20d\n"%( step_number ) )
    fdesc.write( "Print Frequency:      %20d\n"%( print_frequency ) )
    fdesc.write( "Gradient Tolerance:   %20.10lg\n"%( gradient_tolerance ) )
    fdesc.write( "Method:             %22s\n\n"%( method ) )
    fdesc.write( "%10s%20s%20s\n"%( "Step", "Function", "Gradient" ) )
    fdesc.write( "-" * 50 + "\n" )
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
    ndeg = numpy.sqrt( size )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    mol.get_grad()
    grms = numpy.linalg.norm( mol.grad ) / ndeg
    coor = mol.coor[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    grad = mol.grad[sele].ravel().ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    dire = ( ctypes.c_double * size )()
    gold = ( ctypes.c_double * size )()
    work = ( ctypes.c_double * size )()
    iflg = ( ctypes.c_int )()
    fdesc.write( "%30.5lf%20.10lf\n"%( mol.func, grms ) )
    k = 0
    while( k < step_number and grms > gradient_tolerance ):
        dlib.cgp_cgfam_( ctypes.c_int( size ), coor, ctypes.c_double( mol.func ), grad, dire, gold,
                ctypes.c_double( gradient_tolerance ), work, iflg, ctypes.c_int( rest ), ctypes.c_int( meth ) )
        if( iflg == -3 ):
            fdesc.write( "\n -- Improper input parameters...\n" )
            k = step_number + 1
        elif( iflg == -2 ):
            fdesc.write( "\n -- Descent was not obtained...\n" )
            k = step_number + 1
        elif( iflg == -1 ):
            fdesc.write( "\n -- Line Search failure...\n" )
            k = step_number + 1
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
        k = k + 1
        if( k%print_frequency == 0 ):
            fdesc.write( "%10d%20.5lf%20.10lf\n"%( k, mol.func, grms ) )
        mol.current_step( i )
    if( k%print_frequency != 0 ):
        fdesc.write( "%10d%20.5lf%20.10lf\n"%( k + 1, mol.func, grms ) )
    fdesc.write( "-" * 50 + "\n" )
