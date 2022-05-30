#!/usr/bin/env python3
import  sys
import  math
import  numpy
import  typing
import  qm3.utils.grids
import  qm3.utils.interpolation

import  matplotlib.pyplot as plt


def calc_tau( kumb, potm, poti, potp, crdm, crdi, crdp ):
    dcM = crdp - crdi
    dcm = crdi - crdm
    dpM = max( math.fabs( potp - poti ), math.fabs( potm - poti ) )
    dpm = min( math.fabs( potp - poti ), math.fabs( potm - poti ) )
    if( potp > poti and poti > potm ):
        tau = dcM.copy()
    elif( potp < poti and poti < potm ):
        tau = dcm.copy()
    else:
        if( potp > potm ):
            tau = dpM * dcM + dpm * dcm
        else:
            tau = dpm * dcM + dpM * dcm
    tau /= numpy.linalg.norm( tau )
    gum = kumb * numpy.sum( ( dcm - dcM ) * tau ) * tau
    return( tau, gum )


def neb_path( igrd: object, node: int, gues: list, kumb: float ):
    delt = []
    for i in range( 1, len( gues ) ):
        delt.append( numpy.linalg.norm( gues[i] - gues[i-1] ) )
    dtot = sum( delt )
    npts = [ int( round( delt[i] / dtot * ( node + 1 ), 0 ) ) for i in range( len( delt ) ) ]
    delt = []
    for i in range( 1, len( gues ) ):
        delt.append( ( gues[i] - gues[i-1] ) / npts[i-1] )
    npts[-1] += 1
    coor = []
    for i in range( len( gues ) - 1 ):
        for n in range( npts[i] ):
            coor.append( gues[i] + n * delt[i] )
    coor = numpy.array( coor, dtype=numpy.float64 )
    dime = len( coor )
    # ------------------------------------------------------------------------------
    ndeg = math.sqrt( 2.0 * dime )
    snum = 1000
    pfrq = 10
    ssiz = 0.1
    gtol = 0.1 * dime
    nstp = 0
    alph = 0.1
    velo = numpy.zeros( ( dime, 2 ), dtype=numpy.float64 )
    refs = numpy.arange( dime, dtype=numpy.float64 )
    # ------------------------------------------------------------------------------
    func = numpy.zeros( dime, dtype=numpy.float64 )
    grad = numpy.zeros( ( dime, 2 ), dtype=numpy.float64 )
    for i in range( dime ):
        func[i], grad[i,0], grad[i,1] = igrd.calc( coor[i,0], coor[i,1] )
    for i in range( 1, dime - 1 ):
        tau, gum = calc_tau( kumb, func[i-1], func[1], func[i+1], coor[i-1], coor[i], coor[i+1] )
        grad[i] += gum - numpy.sum( tau * grad[i] ) * tau
    norm = numpy.linalg.norm( grad )
    grms = norm / ndeg
    print( "%30.5lf%20.10lf"%( numpy.sum( func ), grms ) )
    itr  = 0
    while( itr < snum and grms > gtol ):
        if( - numpy.sum( velo * grad ) > 0.0 ):
            vsiz = numpy.linalg.norm( velo )
            velo = ( 1.0 - alph ) * velo - alph * grad / norm * vsiz
            if( nstp > 5 ):
                ssiz = min( ssiz * 1.1, 0.01 )
                alph *= 0.99
            nstp += 1
        else:
            alph = 0.1
            ssiz *= 0.5
            nstp = 0
            velo = numpy.zeros( ( dime, 2 ), dtype=numpy.float64 )
        velo -= ssiz * grad
        step = ssiz * velo
        tmp  = numpy.linalg.norm( step )
        if( tmp > ssiz ):
            step *= ssiz / tmp
        coor += step

        obj = qm3.utils.interpolation.gaussian( refs, coor[:,0], 0.5 )
        coor[:,0] = numpy.array( [ obj.calc( i )[0] for i in refs ], dtype=numpy.float64 )
        obj = qm3.utils.interpolation.gaussian( refs, coor[:,1], 0.5 )
        coor[:,1] = numpy.array( [ obj.calc( i )[0] for i in refs ], dtype=numpy.float64 )

        func = numpy.zeros( dime, dtype=numpy.float64 )
        grad = numpy.zeros( ( dime, 2 ), dtype=numpy.float64 )
        for i in range( dime ):
            func[i], grad[i,0], grad[i,1] = igrd.calc( coor[i,0], coor[i,1] )
        for i in range( 1, dime - 1 ):
            tau, gum = calc_tau( kumb, func[i-1], func[1], func[i+1], coor[i-1], coor[i], coor[i+1] )
            grad[i] += gum - numpy.sum( tau * grad[i] ) * tau
        norm = numpy.linalg.norm( grad )
        grms = norm / ndeg

#        plt.clf()
#        grd.plot2d()
#        plt.plot( coor[:,0], coor[:,1], '-o' )
#        plt.savefig( "snap.%04d.png"%( itr ) )

        itr += 1
        if( itr % pfrq == 0 ):
            print( "%10d%20.5lf%20.10lf%20.10lf"%( itr, numpy.sum( func ), grms, ssiz ) )
    if( itr % pfrq != 0 ):
        print( "%10d%20.5lf%20.10lf%20.10lf"%( itr + 1, numpy.sum( func ), grms, ssiz ) )
    return( coor, func )


grd = qm3.utils.grids.grid()
grd.parse( open( sys.argv[1], "rt" ) )
obj = qm3.utils.interpolation.interpolate_2d( grd.x, grd.y, grd.z )
gue = numpy.array( [ float( i ) for i in sys.argv[2:] ], dtype=numpy.float64 )
gue.shape = ( len( gue ) // 2, 2 )
pth, ene = neb_path( obj, 50, gue, 400 )
with open( "path.log", "wt" ) as f:
    for i in range( len( ene ) ):
        f.write( "%20.10lf%20.10lf%20.10lf\n"%( pth[i,0], pth[i,1], ene[i] ) )
