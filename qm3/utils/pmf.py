import  numpy
import  typing
import  math
import  qm3.data


# =================================================================================================

def wham( data_lst: list,
        nbins: typing.Optional[int] = -1,
        temperature: typing.Optional[float] = 300.0,
        nskip: typing.Optional[int] = 0,
        maxit: typing.Optional[int] = 10000,
        toler: typing.Optional[float] = 1.0e-3,
        qprnt: typing.Optional[bool] = True ) -> tuple:
    """
    wham.F90 fDynamo module
    Comput. Phys. Communications v135, p40 (2001) [10.1016/S0010-4655(00)00215-0]
    J. Chem. Theory Comput. v6, p3713 (2010) [10.1021/ct100494z]
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
