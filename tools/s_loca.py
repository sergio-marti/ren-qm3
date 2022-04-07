#!/usr/bin/env python3
import  sys
import  numpy
import  typing
import  qm3.utils.grids
import  qm3.utils.interpolation


def rfo( igrd: object, iniX: float, iniY: float,
        saddle: typing.Optional[bool] = False ):
    # ------------------------------------------------------------------------------
    ndeg = 2
    snum = 1000
    pfrq = 100
    ssiz = 0.01
    gtol = 0.01
    fmod = 0
    coor = numpy.array( [ iniX, iniY ] )
    print( "---------------------------------------- Minimization (RFO)\n" )
    print( "Degrees of Freedom: %20ld"%( ndeg ) )
    print( "Search for saddle:  %20s"%( saddle ) )
    print( "Step Number:        %20d"%( snum ) )
    print( "Step Size:          %20.10lg"%( ssiz ) )
    print( "Print Frequency:    %20d"%( pfrq ) )
    print( "Gradient Tolerance: %20.10lg\n"%( gtol ) )
    print( "%10s%20s%20s"%( "Step", "Function", "Gradient" ) )
    print( "-" * 50 )
    # ------------------------------------------------------------------------------
    tol2 = 1.0e-8
    dx   = numpy.zeros( ndeg )
    dd   = 0.5 * numpy.ones( ndeg )
    if( saddle ):
        dd[fmod] *= -1.0
    grms = gtol * 2.0
    k    = 0
    flg  = True
    while( k < snum and grms > gtol and flg ):
        coor -= dx
        coor[0] = min( max( coor[0], igrd.x[0] ), igrd.x[-1] )
        coor[1] = min( max( coor[1], igrd.y[0] ), igrd.y[-1] )

        func, gx, gy = igrd.calc( coor[0], coor[1] )
        grad = numpy.array( [ gx, gy ] )
        dh = 1.e-6
        of = igrd.calc( coor[0] + dh, coor[1] )
        ob = igrd.calc( coor[0] - dh, coor[1] )
        r1 = [ ( of[1] - ob[1] ) / ( 2. * dh ), ( of[2] - ob[2] ) / ( 2. * dh ) ]
        of = igrd.calc( coor[0], coor[1] + dh )
        ob = igrd.calc( coor[0], coor[1] - dh )
        r2 = [ ( of[1] - ob[1] ) / ( 2. * dh ), ( of[2] - ob[2] ) / ( 2. * dh ) ]
        hess = numpy.array( [ [ r1[0], ( r1[1] + r2[0] ) / 2. ], [ ( r1[1] + r2[0] ) / 2., r2[1] ] ] )

        val, vec = numpy.linalg.eigh( hess )
        idx = numpy.argsort( val )
        val = val[idx]
        vec = vec[:,idx]

        vg  = numpy.dot( vec.T, grad )
        lg  = dd * ( numpy.fabs( val ) + numpy.sqrt( val * val + 4 * vg * vg ) )
        dx  = numpy.dot( vec, vg / lg )
        tt  = numpy.linalg.norm( dx )
        if( tt < tol2 ):
            flg = False
        if( tt > ssiz ):
            dx *= ssiz / tt

        k   += 1
        grms = numpy.sqrt( numpy.sum( grad * grad ) / 2 )
        if( k % pfrq == 0 ):
            print( "%10ld%20.5lf%20.10lf"%( k, func, grms ) )
    
    if( k % pfrq != 0 ):
        print( "%10ld%20.5lf%20.10lf"%( k, func, grms ) )
    print( "-" * 50 )
    # ------------------------------------------------------------------------------
    print( coor[0], coor[1], func )



grd = qm3.utils.grids.grid()
grd.parse( open( sys.argv[1], "rt" ) )
rfo( qm3.utils.interpolation.interpolate_2d( grd.x, grd.y, grd.z ),
        float( sys.argv[2] ),
        float( sys.argv[3] ),
        int( sys.argv[4] ) == 1 )
