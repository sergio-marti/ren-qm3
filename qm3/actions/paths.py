import  sys
import  math
import  numpy
import  typing
import  qm3.utils
import  qm3.utils.hessian


__vcut = 0.00035481432270250985 # 1 _cm^-1


def initial_step( mol: object, get_hess: typing.Callable, step_size: float ) -> tuple:
    """
    returns: the number of 'negative' eigen-values (see __vcut) and the mass-weighted transition mode
    """
    actv = numpy.sum( mol.actv )
    size = 3 * actv
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    mass = 1.0 / numpy.sqrt( mol.mass[sele] )
    hess = get_hess( mol, 0 )
    for i in range( size ):
        for j in range( size ):
            hess[i,j] *= mass[i//3] * mass[j//3]
    val, vec = numpy.linalg.eigh( qm3.utils.hessian.project_RT( hess, qm3.utils.RT_modes( mol ) ) )
    idx = numpy.argsort( val )
    vec = vec[:,idx]
    return( numpy.sum( val < __vcut ), step_size * vec[:,0].reshape( ( actv, 3 ) ) / numpy.linalg.norm( vec[:,0] ) * mass )


def page_mciver( mol: object,
        get_hess: typing.Callable,
        step_number: typing.Optional[int] = 100,
        step_size: typing.Optional[float] = 0.01,
        print_frequency: typing.Optional[int] = 10,
        gradient_tolerance: typing.Optional[float] = 1.5,
        from_saddle: typing.Optional[bool] = True,
        fdsc: typing.Optional[typing.IO] = sys.stdout ):
    """
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
    fdsc.write( "---------------------------------------- Minimum Path (Page-McIver:LQA)\n\n" )
    fdsc.write( "Degrees of Freedom: %20ld\n"%( size ) )
    fdsc.write( "Step Number:        %20d\n"%( step_number ) )
    fdsc.write( "Step Size:          %20.10lg\n"%( step_size ) )
    fdsc.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    fdsc.write( "Gradient Tolerance: %20.10lg\n"%( gradient_tolerance ) )
    fdsc.write( "From Saddle:        %20s\n\n"%( from_saddle ) )
    fdsc.write( "%10s%20s%20s%10s\n"%( "Step", "Function", "Gradient", "Nneg" ) )
    fdsc.write( "-" * 60 + "\n" )
    it2m = 1000
    it3m = 100000
    ndeg = math.sqrt( size )
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    grms = gradient_tolerance * 2.0
    mass = numpy.sqrt( mol.mass[sele] )
    if( from_saddle ):
        nskp, dsp = initial_step( mol, get_hess, step_size )
        fdsc.write( "%10s%20.5lf%20.10lf%10ld\n"%( "", mol.func, numpy.linalg.norm( mol.grad ) / ndeg, nskp ) )
    else:
        nskp = 7
        dsp  = numpy.zeros( ( actv, 3 ), dtype=numpy.float64 )
    ssiz = math.fabs( step_size )
    it1 = 0
    flg = True
    while( it1 < step_number and ( grms > gradient_tolerance or nskp > 6 ) and flg ):
        mol.coor[sele] += dsp
        hes = get_hess( mol, it1 )
        for i in range( size ):
            for j in range( size ):
                hes[i,j] /= mass[i//3] * mass[j//3]
        mod = qm3.utils.RT_modes( mol )
        val, vec = numpy.linalg.eigh( qm3.utils.hessian.project_RT( hes, mod ) )
        idx = numpy.argsort( val )
        val = val[idx]
        vec = vec[:,idx]
        nskp = numpy.sum( val < __vcut )
        grd = mol.grad[sele] / mass
        grd.shape = ( size, )
        for i in range( 6 ):
            grd -= numpy.sum( grd * mod[i,:] ) * mod[i,:]
        tmp = numpy.linalg.norm( grd )
        grd /= tmp
        val /= tmp
        grd = numpy.dot( vec.T, grd ) 
        # -- LQA step ----------------------------------------------------------------
        pm_dt = 0.2 * ssiz
        pm_t  = 1.e10
        pm_ot = 0.0
        it2   = 0
        it3   = 0
        while( math.fabs( 1.0 - pm_ot / pm_t ) > 1.0e-6 and it2 < it2m and it3 < it3m ):
            it2  += 1
            pm_ot = pm_t
            pm_dt = 0.5 * pm_dt
            pm_t  = 0.0
            pm_ft = math.sqrt( numpy.sum( numpy.square( grd * numpy.exp( - val * pm_t ) ) ) )
            pm_s  = 0.0;
            it3   = 0;
            while( pm_s < ssiz and it3 < it3m ):
                it3  += 1
                pm_os = pm_s
                pm_of = pm_ft
                pm_t += pm_dt
                pm_ft = math.sqrt( numpy.sum( numpy.square( grd * numpy.exp( - val * pm_t ) ) ) )
                pm_s += 0.5 * pm_dt * ( pm_ft + pm_of )
            if( pm_os != pm_s ):
                pm_t -= ( ssiz - pm_s ) * pm_dt / ( pm_os - pm_s )
            else:
                fdsc.write( "\n -- The current step-size did not converge...\n" )
                flg = False
        if( math.fabs( 1.0 - pm_t / pm_ot ) <= 1.0e-6 and flg ):
            for i in range( size ):
                tmp = val[i] * pm_t
                if( math.fabs( tmp ) < 1.e-8 ):
                    grd[i] *= - pm_t * ( 1.0 - tmp  * ( 0.5 - tmp / 6.0 ) )
                else:
                    grd[i] *= ( math.exp( - tmp ) - 1.0 ) / val[i]
            dsp = numpy.dot( vec, grd )
            dsp.shape = ( actv, 3 )
            dsp /= mass
        # ----------------------------------------------------------------------------
        it1 += 1
        grms = numpy.linalg.norm( mol.grad ) / ndeg
        if( it1 % print_frequency == 0 ):
            fdsc.write( "%10ld%20.5lf%20.10lf%10ld\n"%( it1, mol.func, grms, nskp ) )
        mol.current_step( it1 )
    if( it1 % print_frequency != 0 ):
        fdsc.write( "%10ld%20.5lf%20.10lf%10ld\n"%( it1, mol.func, grms, nskp ) )
    fdsc.write( "-" * 60 + "\n" )
