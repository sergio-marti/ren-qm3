import  numpy
import  typing
import  math
import  qm3.data


# =================================================================================================

def wham( data_lst: list,
        nskip: typing.Optional[int] = 0,
        nbins: typing.Optional[int] = -1,
        temperature: typing.Optional[float] = 300.0,
        maxit: typing.Optional[int] = 10000,
        toler: typing.Optional[float] = 1.0e-3 ) -> tuple:
    """
    wham.F90 fDynamo module
    Comput. Phys. Communications v135, p40 (2001) [doi:10.1016/S0010-4655(00)00215-0]
    J. Chem. Theory Comput. v6, p3713 (2010) [doi:10.1021/ct100494z]
    """
    # ---------------------------------------------
    data = []
    nwin = len( data_lst )
    kumb = numpy.zeros( nwin, dtype=numpy.float64 )
    xref = numpy.zeros( nwin, dtype=numpy.float64 )
    ndat = numpy.zeros( nwin, dtype=numpy.float64 )
    for k in range( nwin ):
        with open( data_lst[k], "rt" ) as f:
            kumb[k], xref[k] = ( float( i ) for i in f.readline().strip().split() )
            i = 0
            for l in f:
                if( i >= nskip ):
                    data.append( float( l.strip() ) )
                    ndat[k] += 1.0
                i += 1
    data = numpy.array( data, dtype=numpy.float64 )
    # ---------------------------------------------
    if( nbins < nwin ):
        bins = nwin * 2
    else:
        bins = nbins
    minx = numpy.min( data )
    maxx = numpy.max( data )
    dbin = ( maxx - minx ) / bins
    coor = numpy.linspace( minx + dbin * 0.5, maxx - dbin * 0.5, bins )
    freq = numpy.zeros( bins, dtype=numpy.float64 )
    for i in range( len( data ) ):
        freq[min( max( int( ( data[i] - minx ) / dbin ), 0 ), bins - 1 )] += 1.0
    del data
    # ---------------------------------------------
    rt  = temperature * 1.0e-3 * qm3.data.R
    rho = numpy.zeros( bins, dtype=numpy.float64 )
    umb = numpy.zeros( ( bins, nwin ), dtype=numpy.float64 )
    for i in range( bins ):
        umb[i,:] = 0.5 * kumb * numpy.square( coor[i] - xref )
    emb = numpy.exp( - umb / rt )
    frc = numpy.zeros( nwin, dtype=numpy.float64 )
    rho = numpy.zeros( bins, dtype=numpy.float64 )
    itr = 0
    flg = False
    while( itr < maxit and not flg ):
        bak = frc.copy()
        for i in range( bins ):
            rho[i] = freq[i] / numpy.sum( ndat * numpy.exp( - ( umb[i,:] - frc ) / rt ) )
        for j in range( nwin ):
            frc[j] = - rt * math.log( numpy.sum( emb[:,j] * rho ) )
        flg = numpy.max( numpy.fabs( bak - frc ) ) < toler
        itr += 1
    # ---------------------------------------------
    print( "[wham]%10s%10d"%( flg, itr ) )
    if( flg ):
        rho /= numpy.sum( rho )
        func = - rt * numpy.log( numpy.where( rho > 0.0, rho, 1.0 ) )
    else:
        func = numpy.zeros( bins )
    return( coor, func - numpy.max( func ) )

# =================================================================================================

def umbint( data_lst: list,
        nskip: typing.Optional[int] = 0,
        nbins: typing.Optional[int] = -1,
        temperature: typing.Optional[float] = 300.0 ) -> tuple:
    """
    J. Chem. Phys. v123, p144104 (2005) [doi:10.1063/1.2052648]
    J. Chem. Phys. v124, p234106 (2006) [doi:10.1063/1.2206775]
    """
    # ---------------------------------------------
    def __dAdx( x: float, rt: float, gs: float,
            frq: numpy.array, avr: numpy.array, sdv: numpy.array,
            kmb: numpy.array, ref: numpy.array ) -> float:
        den = frq * gs / sdv * numpy.exp( - 0.5 * numpy.square( ( x - avr ) / sdv ) )
        num = den * ( rt * ( x - avr ) / numpy.square( sdv ) - kmb * ( x - ref ) )
        return( numpy.sum( num ) / numpy.sum( den ) )

    # from eq. 5: var( mu ) = sig^2 / n; var( sig^2 ) = ( sum ( x - mu )^4 / n - sig^4 ) / n
    def __vdAdx( x: float, rt: float, gs: float,
            frq: numpy.array, avr: numpy.array, sdv: numpy.array,
            sd2: numpy.array, sd4: numpy.array, var: numpy.array ) -> float:
        den = frq * gs / sdv * numpy.exp( - 0.5 * numpy.square( ( x - avr ) / sdv ) )
        num = numpy.square( den ) * rt * rt * ( 1.0 / sd2 + numpy.square( x - avr ) * ( var - sd4 ) / sd4 ) / frq
        den = numpy.sum( den )
        return( numpy.sum( num ) / ( den * den ) )
    # ---------------------------------------------
    nwin = len( data_lst )
    kumb = numpy.zeros( nwin, dtype=numpy.float64 )
    xref = numpy.zeros( nwin, dtype=numpy.float64 )
    ndat = numpy.zeros( nwin, dtype=numpy.float64 )
    aver = numpy.zeros( nwin, dtype=numpy.float64 )
    sdev = numpy.zeros( nwin, dtype=numpy.float64 )
    vsig = numpy.zeros( nwin, dtype=numpy.float64 )
    minx = None
    maxx = None
    for k in range( nwin ):
        with open( data_lst[k], "rt" ) as f:
            kumb[k], xref[k] = ( float( i ) for i in f.readline().strip().split() )
            s3 = 0.0
            s4 = 0.0
            i  = 0
            for l in f:
                if( i >= nskip ):
                    t = float( l.strip() )
                    if( minx == None ):
                        minx = t
                        maxx = t
                    else:
                        minx = min( minx, t )
                        maxx = max( maxx, t )
                    ndat[k] += 1.0
                    aver[k] += t
                    sdev[k] += t * t
                    t2       = t * t
                    s3      += t2 * t
                    s4      += t2 * t2
                i += 1
            aver[k] /= ndat[k]
            s2       = sdev[k] / ndat[k]
            sdev[k]  = math.sqrt( math.fabs( sdev[k] / ndat[k] - aver[k] * aver[k] ) )
            s3      /= ndat[k]
            s4      /= ndat[k]
            m2       = aver[k] * aver[k]
            vsig[k]  = s4 + 3.0 * m2 * ( 2.0 * s2 - m2 ) - 4.0 * aver[k] * s3
    # ---------------------------------------------
    if( nbins < nwin ):
        bins = nwin * 2
    else:
        bins = nbins
    dbin = ( maxx - minx ) / bins
    coor = numpy.linspace( minx + dbin * 0.5, maxx - dbin * 0.5, bins )
    gs   = 1.0 / math.sqrt( 2.0 * math.pi )
    rt   = temperature * 1.0e-3 * qm3.data.R
    # ---------------------------------------------
    func = numpy.zeros( bins )
    e    = 0.0
    l    = __dAdx( coor[0], rt, gs, ndat, aver, sdev, kumb, xref )
    for i in range( 1, bins ):
        e       = __dAdx( coor[i], rt, gs, ndat, aver, sdev, kumb, xref )
        func[i] = func[i-1] + 0.5 * dbin * ( l + e )
        l       = e
    # ---------------------------------------------
    ferr = numpy.zeros( bins )
    sd2  = numpy.square( sdev )
    sd4  = numpy.square( sd2 )
    e    = 0.0
    l    = __vdAdx( coor[0], rt, gs, ndat, aver, sdev, sd2, sd4, vsig )
    for i in range( 1, bins ):
        e       = __vdAdx( coor[i], rt, gs, ndat, aver, sdev, sd2, sd4, vsig )
        ferr[i] = ferr[i-1] + 0.5 * dbin * ( l + e )
        l       = e
    # ---------------------------------------------
    return( coor, func - numpy.max( func ), numpy.sqrt( ferr ) )
