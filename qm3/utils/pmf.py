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
#        with open( data_lst[k], "rt" ) as f:
#            kumb[k], xref[k] = ( float( i ) for i in f.readline().strip().split() )
#            i = 0
#            for l in f:
#                if( i >= nskip ):
#                    data.append( float( l.strip() ) )
#                    ndat[k] += 1.0
#                i += 1
    # ---------------------------------------------
        kumb[k], xref[k] = ( float( i ) for i in data_lst[k].readline().strip().split() )
        i = 0
        for l in data_lst[k]:
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
#        with open( data_lst[k], "rt" ) as f:
#            kumb[k], xref[k] = ( float( i ) for i in f.readline().strip().split() )
#            s3 = 0.0
#            s4 = 0.0
#            i  = 0
#            for l in f:
#                if( i >= nskip ):
#                    t = float( l.strip() )
#                    if( minx == None ):
#                        minx = t
#                        maxx = t
#                    else:
#                        minx = min( minx, t )
#                        maxx = max( maxx, t )
#                    ndat[k] += 1.0
#                    aver[k] += t
#                    sdev[k] += t * t
#                    t2       = t * t
#                    s3      += t2 * t
#                    s4      += t2 * t2
#                i += 1
#            aver[k] /= ndat[k]
#            s2       = sdev[k] / ndat[k]
#            sdev[k]  = math.sqrt( math.fabs( sdev[k] / ndat[k] - aver[k] * aver[k] ) )
#            s3      /= ndat[k]
#            s4      /= ndat[k]
#            m2       = aver[k] * aver[k]
#            vsig[k]  = s4 + 3.0 * m2 * ( 2.0 * s2 - m2 ) - 4.0 * aver[k] * s3
    # ---------------------------------------------
        kumb[k], xref[k] = ( float( i ) for i in data_lst[k].readline().strip().split() )
        s3 = 0.0
        s4 = 0.0
        i  = 0
        for l in data_lst[k]:
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
                t2       = t * t
                sdev[k] += t2
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

# =================================================================================================

def average_force_integration( data_lst: list,
        nskip: typing.Optional[int] = 0,
        use_sampling_ratio: typing.Optional[bool] = False ) -> tuple:
    """
    J. Comput. Chem. v33, p435 (2012) [doi:10.1002/jcc.21989]


∆G_{ i rightarrow i+1 } approx { 1 over 2 } left lbrace k_{i+1} ( r_{i+1} - langle x_{i+1} rangle ) + k_i ( r_i - langle x_i rangle ) right rbrace cdot left( langle x_{i+1} rangle - langle x_i rangle  right )
newline newline
J_{i,i+1} = left( {partial ∆G_{ i rightarrow i+1 }} over{partial langle x_i rangle}, {∆G_{ i rightarrow i+1 }}over{partial langle x_{i+1} rangle} right )_{ 1x2 }
newline newline
C_{ i,i+1 } = left(
matrix{
{1+ 2%tau_i} over N_i sum{ (x_i - langle x_i rangle )^2} #
sqrt{(1+ 2%tau_i)(1+ 2%tau_{i+1})} over sqrt{N_i N_{i+1}} sum{ (x_i - langle x_i rangle ) (x_{i+1} - langle x_{i+1} rangle )} ##
dotsup #
{1+ 2%tau_{i+1}} over N_{i+1} sum{ (x_{i+1} - langle x_{i+1} rangle )^2}
} right )_{ 2x2 }
newline newline
s^2( ∆G_{ i rightarrow i+1 } ) = J_{i,i+1} cdot C_{ i,i+1 } cdot J_{i,i+1}^T
newline newline
1 + 2 %tau = { 1 + r_1 } over { 1 - r_1 }
~~~~~~~~
r_1 =  sum from{2} to{N}{ (x_i - langle x_i rangle ) (x_{i-1} - langle x_i rangle ) } over sum{ (x_i - langle x_i rangle )^2 }
    """
    # ---------------------------------------------
    nwin = len( data_lst )
    kumb = numpy.zeros( nwin, dtype=numpy.float64 )
    xref = numpy.zeros( nwin, dtype=numpy.float64 )
    aver = numpy.zeros( nwin, dtype=numpy.float64 )
    vari = numpy.zeros( nwin, dtype=numpy.float64 )
    srat = numpy.zeros( nwin, dtype=numpy.float64 )
    data = []
    for k in range( nwin ):
        kumb[k], xref[k] = ( float( i ) for i in data_lst[k].readline().strip().split() )
        data.append( [] )
        i  = 0
        for l in data_lst[k]:
            if( i >= nskip ):
                data[-1].append( float( l.strip() ) )
            i += 1
        data[-1] = numpy.array( data[-1] )
        aver[k]  = data[-1].mean()
        vari[k]  = data[-1].var() * data[-1].shape[0]
        tmp      = data[-1] - aver[k]
        cor      = sum( [ tmp[i] * ( data[-1][i+1] - aver[k] ) for i in range( data[-1].shape[0] - 1 ) ] ) / vari[k] 
        srat[k]  = ( 1.0 + cor ) / ( 1.0 - cor )
    # ---------------------------------------------
    if( not use_sampling_ratio ):
        srat = numpy.ones( nwin, dtype=numpy.float64 )
    indx = numpy.argsort( xref )
    coor = numpy.zeros( nwin - 1 )
    func = numpy.zeros( nwin - 1 )
    ferr = numpy.zeros( nwin - 1 )
    for i in range( indx.shape[0] - 1 ):
        coor[i] = 0.5 * ( aver[indx[i]] + aver[indx[i+1]] )
        # ---------------------------------------------
        # averages instead of the references in eq. 28 (it also affects the jacobian)
        #func[i] = kumb[indx[i+1]] * ( xref[indx[i+1]] - aver[indx[i+1]] ) + kumb[indx[i]] * ( xref[indx[i]] - aver[indx[i]] )
        #func[i] *= 0.5 * ( aver[indx[i+1]] - aver[indx[i]] )
        #jaco = [ .0, .0 ]
        #jaco[0] = - 0.5 * ( kumb[indx[i]] * xref[indx[i]] + kumb[indx[i+1]] * xref[indx[i+1]] )
        #jaco[0] +=   kumb[indx[i]]   * aver[indx[i]]   + 0.5 * aver[indx[i+1]] * ( kumb[indx[i+1]] - kumb[indx[i]] )
        #jaco[1] =   0.5 * ( kumb[indx[i]] * xref[indx[i]] + kumb[indx[i+1]] * xref[indx[i+1]] )
        #jaco[1] += - kumb[indx[i+1]] * aver[indx[i+1]] + 0.5 * aver[indx[i]]   * ( kumb[indx[i+1]] - kumb[indx[i]] )
        # ---------------------------------------------
        func[i] = kumb[indx[i+1]] * ( xref[indx[i+1]] - aver[indx[i+1]] ) + kumb[indx[i]] * ( xref[indx[i]] - aver[indx[i]] )
        func[i] *= 0.5 * ( xref[indx[i+1]] - xref[indx[i]] )
        jaco = [ 0.5 * kumb[indx[i]] * ( xref[indx[i]] - xref[indx[i+1]] ),
                 0.5 * kumb[indx[i+1]] * ( xref[indx[i]] - xref[indx[i+1]] ) ]
        # ---------------------------------------------
        ndat = min( data[indx[i]].shape[0], data[indx[i+1]].shape[0] )
        mcov = [ vari[indx[i]] * srat[indx[i]] / ndat, 0.0, vari[indx[i+1]] * srat[indx[i+1]] / ndat ]
        for j in range( ndat ):
            mcov[1] += ( data[indx[i]][j] - aver[indx[i]] ) * ( data[indx[i+1]][j] - aver[indx[i+1]] )
        mcov[1] *= math.sqrt( srat[indx[i]] * srat[indx[i+1]] ) / ndat
        ferr[i] = math.sqrt( jaco[0] * ( mcov[0] * jaco[0] + mcov[1] * jaco[1] ) + jaco[1] * ( mcov[1] * jaco[0] + mcov[2] * jaco[1] ) )
    # ---------------------------------------------
    func = numpy.cumsum( func )
    return( coor, func - numpy.max( func ), ferr )
