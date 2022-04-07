#!/usr/bin/env python3
import  sys
import  math
import  numpy
import  typing
import  qm3.utils.grids
import  qm3.utils.interpolation


def taylor( igrd: object, iniX: float, iniY: float, direc: float, fdsc: typing.IO ):
    # ------------------------------------------------------------------------------
    ndeg = 2
    snum = 1000
    pfrq = 100
    ssiz = 0.01
    gtol = 0.01
    coor = numpy.array( [ iniX, iniY ] )
    print( "---------------------------------------- Minimum Path (Taylor)\n" )
    print( "Degrees of Freedom: %20ld"%( ndeg ) )
    print( "Step Number:        %20d"%( snum ) )
    print( "Step Size:          %20.10lg"%( ssiz * direc ) )
    print( "Print Frequency:    %20d"%( pfrq ) )
    print( "Gradient Tolerance: %20.10lg\n"%( gtol ) )
    print( "%10s%20s%20s"%( "Step", "Function", "Gradient" ) )
    print( "-" * 50 )
    # ------------------------------------------------------------------------------
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
    step = ssiz * direc * vec[:,0] / numpy.linalg.norm( vec[:,0] )
    k    = 0
    flg  = True
    while( k < snum and flg ):
        coor[0] = min( max( coor[0] + step[0], igrd.x[0] ), igrd.x[-1] )
        coor[1] = min( max( coor[1] + step[1], igrd.y[0] ), igrd.y[-1] )
        last = func
        func, gx, gy = igrd.calc( coor[0], coor[1] )
        fdsc.write( "%20.10lf%20.10lf%20.4lf\n"%( coor[0], coor[1], func ) )
        flg  = math.fabs( last - func ) > 1.e-6
        grad = numpy.array( [ gx, gy ] )
        norm = numpy.linalg.norm( grad )
        dh = 1.e-6
        of = igrd.calc( coor[0] + dh, coor[1] )
        ob = igrd.calc( coor[0] - dh, coor[1] )
        r1 = [ ( of[1] - ob[1] ) / ( 2. * dh ), ( of[2] - ob[2] ) / ( 2. * dh ) ]
        of = igrd.calc( coor[0], coor[1] + dh )
        ob = igrd.calc( coor[0], coor[1] - dh )
        r2 = [ ( of[1] - ob[1] ) / ( 2. * dh ), ( of[2] - ob[2] ) / ( 2. * dh ) ]
        hess = numpy.array( [ [ r1[0], ( r1[1] + r2[0] ) / 2. ], [ ( r1[1] + r2[0] ) / 2., r2[1] ] ] )
        # Eqs 2, 4, 7 & 13 of J. Chem. Phys. v88, p922 (1988) [10.1063/1.454172]
        v0   = - grad / norm
        tt   = numpy.dot( hess, v0.reshape( ( 2, 1 ) ) ).ravel()
        pp   = numpy.sum( v0 * tt )
        v1   = ( tt - pp * v0 ) / norm
        step = ssiz * ( v0 + 0.5 * ssiz * v1 )
        k   += 1
        grms = norm / math.sqrt( 2.0 )
        if( k % pfrq == 0 ):
            print( "%10ld%20.5lf%20.10lf"%( k, func, grms ) )
    if( k % pfrq != 0 ):
        print( "%10ld%20.5lf%20.10lf"%( k, func, grms ) )
    print( "-" * 50 )



grd = qm3.utils.grids.grid()
grd.parse( open( sys.argv[1], "rt" ) )
inX = float( sys.argv[2] )
inY = float( sys.argv[3] )
obj = qm3.utils.interpolation.interpolate_2d( grd.x, grd.y, grd.z )
log = open( "path.log", "wt" )
log.write( "# forward\n" )
taylor( obj, inX, inY,  1.0, log )
log.write( "# reverse\n" )
taylor( obj, inX, inY, -1.0, log )
log.close()
