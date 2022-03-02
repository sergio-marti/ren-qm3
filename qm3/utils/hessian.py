import  numpy
import  typing
import  qm3.data


def numerical( mol: object,
        dsp: typing.Optional[float] = 1.e-4, central: typing.Optional[bool] = True ) -> numpy.array:
    size = 3 * mol.actv.sum()
    sele = numpy.argwhere( mol.actv.ravel() )
    hess = numpy.zeros( ( size, size ), dtype=numpy.float64 )
    if( central ):
        k = 0
        for i in sele:
            for j in [0, 1, 2]:
                bak = mol.coor[i,j]
                mol.coor[i,j] = bak + dsp
                mol.get_grad()
                gp = mol.grad[sele].ravel()
                mol.coor[i,j] = bak - dsp
                mol.get_grad()
                hess[k,:] = ( gp - mol.grad[sele].ravel() ) / ( 2.0 * dsp )
                mol.coor[i,j] = bak
                k += 1
    else:
        mol.get_grad()
        ref = mol.grad[sele].ravel()
        k = 0
        for i in range( mol.size ):
            for j in [0, 1, 2]:
                mol.coor[i,j] += dsp
                mol.get_grad()
                hess[k,:] = ( mol.grad[sele].ravel() - ref ) / ( 2.0 * dsp )
                mol.coor[i,j] -= dsp
                k += 1
    # symmetrize
    hess = 0.5 * ( hess + hess.T )
    return( hess )


def RT_modes( mol: object ) -> numpy.array:
    size = 3 * mol.actv.sum()
    sele = numpy.argwhere( mol.actv.ravel() )
    mode = numpy.zeros( ( 6, size ) )
    cent = numpy.sum( mol.mass * mol.coor * mol.actv, axis = 0 ) / numpy.sum( mol.mass * mol.actv )
    k = 0
    for i in sele:
        sqrm = numpy.sqrt( mol.mass[i] )
        mode[0,k:k+3] = [ sqrm, 0.0, 0.0 ]
        mode[1,k:k+3] = [ 0.0, sqrm, 0.0 ]
        mode[2,k:k+3] = [ 0.0, 0.0, sqrm ]
        mode[3,k:k+3] = [ 0.0, - ( mol.coor[i,2] - cent[2] ) * sqrm, ( mol.coor[i,1] - cent[1] ) * sqrm ]
        mode[4,k:k+3] = [ ( mol.coor[i,2] - cent[2] ) * sqrm, 0.0, - ( mol.coor[i,0] - cent[0] ) * sqrm ]
        mode[5,k:k+3] = [ - ( mol.coor[i,1] - cent[1] ) * sqrm, ( mol.coor[i,0] - cent[0] ) * sqrm, 0.0 ]
        k += 3
    # orthogonalize modes
    for i in range( 6 ):
        for j in range( i ):
            mode[i] -= numpy.sum( mode[i] * mode[j] ) * mode[j]
        tmp = numpy.sqrt( numpy.sum( mode[i] * mode[i] ) )
        if( tmp > 0.0 ):
            mode[i] /= tmp
    return( mode )


def project_RT( hess: numpy.array, mode: numpy.array ) -> numpy.array:
    size = hess.shape[0]
# -----------------------------------------------------
#    # G' = G - Tx * G * Tx - ... - Rx * G * Rx - ...
#    for i in range( 6 ):
#        grad -= numpy.sum( grad * rt[i] ) * rt[i]
# -----------------------------------------------------
    # P = I - Tx * Tx - ... - Rx * Rx - ... 
    proj = numpy.identity( size )
    for i in range( size ):
        for j in range( size ):
            for l in range( 6 ):
                proj[i,j] -= mode[l,i] * mode[l,j]
    # H' = P * H * P
    return( numpy.dot( proj, numpy.dot( hess, proj ) ) )


def frequencies( mol: object, hess: numpy.array, project: typing.Optional[bool] = True ) -> tuple:
    size = 3 * mol.actv.sum()
    mass = 1.0 / numpy.sqrt( mol.mass[mol.actv] )
    temp = hess.copy()
    for i in range( size ):
        for j in range( size ):
            temp[i,j] *= mass[i//3] * mass[j//3]
    if( project ):
        temp = project_RT( temp, RT_modes( mol ) )
    freq, mods = numpy.linalg.eigh( temp )
    sidx = numpy.argsort( freq )
    freq = freq[sidx]
    mods = mods[:,sidx]
    wns = 1.0e11 / ( 2. * numpy.pi * qm3.data.C )
    for i in range( size ):
        if( freq[i] < 0.0 ):
            freq[i] = - numpy.sqrt( numpy.fabs( freq[i] ) ) * wns
        else:
            freq[i] =   numpy.sqrt( numpy.fabs( freq[i] ) ) * wns
        for j in range( size ):
            mods[i,j] *= mass[j//3]
    # freq: cm^-1, mods: 1/sqrt[g/mol]
    return( freq, mods )


def IR_intensities( mol: object, mode: numpy.array ) -> numpy.array:
    actv = mol.actv.sum()
    size = 3 * actv
    chrg = mol.chrg[mol.actv.ravel()]
    chrg = numpy.column_stack( ( chrg, chrg, chrg ) ).reshape( size )
    inte = numpy.zeros( size )
    for i in range( 6, size ):
        temp = chrg * mode[:,i]
        temp.shape = ( actv, 3 )
        temp = numpy.sum( temp, axis=0 )
        inte[i] = numpy.dot( temp, temp )
    # inte: km/mol [Na^2 * qe^2 / ( 12 * eps0 * c^2 )]
    return( inte * 974.8802240597 )


try:
    import matplotlib.pyplot
    # -----------------------------------------------
    # adjust frequency values:
    # https://cccbdb.nist.gov/vibscalejust.asp
    # -----------------------------------------------
    # https://en.wikipedia.org/wiki/Spectral_line_shape
    # Lorentzian: L = 1 over { 1 + x^2 } ~~~~~~ x = { p^0 - p } over { s / 2 }
    # -----------------------------------------------
    def IR_spectrum( freq: numpy.array, inte: numpy.array,
            sigm: typing.Optional[float] = 100.,
            minf: typing.Optional[float] = 100.,
            maxf: typing.Optional[float] = 4000.,
            scal: typing.Optional[float] = 1.0 ):
        nn = len( freq )
        hs = 0.5 * sigm
        sx = numpy.arange( int( minf ), int( maxf ), dtype=numpy.float64 )
        sy = numpy.zeros( len( sx ), dtype=numpy.float64 )
        for i in range( nn ):
            if( freq[i] >= minf ):
                sy += inte[i] / ( 1.0 + numpy.square( ( freq[i] * scal - sx ) / hs ) )
        sy /= numpy.max( sy )
        matplotlib.pyplot.clf()
        matplotlib.pyplot.grid( True )
        matplotlib.pyplot.xlim( maxf + sigm, minf - sigm )
        matplotlib.pyplot.ylim( 1.05, -0.05 )
        matplotlib.pyplot.plot( sx, sy, '-' )
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig( "spectrum.pdf" )
except:
    pass



def force_constants( mol: object, freq: numpy.array, mods: numpy.array ) -> tuple:
    siz = 3 * mol.actv.sum()
    # eigenvalues: cm^-1 >> kJ/g.A^2
    val = numpy.square( freq * 2.0 * numpy.pi * qm3.data.C / 1.0e11 )
    # eigenvectors: dimensionless
    mas = mol.mass[mol.actv]
    mas = numpy.column_stack( ( mas, mas, mas ) ).reshape( siz )
    vec = numpy.sqrt( mas ) * mods
    # reduced masses: g/mol
    #rmas = numpy.array( [ 1.0 / sum( [ vec[i,j] * vec[i,j] / mas[i//3] for i in range( size ) ] ) for j in range( size ) ] )
    rmas = 1.0 / numpy.sum( numpy.square( vec.T ) / mas, axis = 1 )
    # force qm3. mDyne/A
    frce = 1.e21 / qm3.data.NA * rmas * val
    return( rmas, frce )



def normal_mode( mol: object, freq: numpy.array, mods: numpy.array, who: int,
        temp: typing.Optional[float] = 298.15,
        afac: typing.Optional[float] = 1.0 ):
    siz = mol.actv.sum()
    sel = numpy.argwhere( mol.actv.ravel() ).ravel()
    ome = numpy.fabs( freq[who] )
    if( ome < 0.1 ): 
        return
    wns = 1.0e11 / ( 2. * numpy.pi * qm3.data.C )
    amp = numpy.sqrt( 2. * 1.0e-3 * qm3.data.KB * qm3.data.NA * temp ) * ( wns / ome )
    fd = open( "nmode.%d"%( who ), "wt" )
    for i in range( 10 ):
        for j in range( 10 ):
            fd.write( "%d\n%10.3lf cm^-1\n"%( siz, freq[who] ) )
            fac = afac * amp * numpy.sin( 2. * numpy.pi * float(j) / 10. )
            for k in range( siz ):
                k3 = 3 * k
                fd.write( "%-4s%20.12lf%20.12lf%20.12lf\n"%(
                    qm3.data.symbol[mol.anum[sel[k]]],
                    mol.coor[sel[k],0] + fac * mods[k3,who],
                    mol.coor[sel[k],1] + fac * mods[k3,who],
                    mol.coor[sel[k],2] + fac * mods[k3,who] ) )
    fd.close()



def rrho( mol: object, freq: numpy.array,
        temp: typing.Optional[float] = 298.15,
        pres: typing.Optional[float] = 1.0,
        symm: typing.Optional[float] = 1.0,
        fcut: typing.Optional[float] = 10.0 ) -> tuple:
    size = mol.actv.sum()
    sele = numpy.argwhere( mol.actv.ravel() ).ravel()
    mass = mol.mass[mol.actv].ravel()
    # Translation (divided by N_Avogadro)
    mt = numpy.sum( mol.mass * mol.actv )
    qt = numpy.power( 2.0 * numpy.pi * mt / qm3.data.NA * 1.0e-3 * qm3.data.KB * temp / ( qm3.data.H * qm3.data.H ), 1.5 ) * qm3.data.KB * temp / ( pres * 1.013250E+5 )
    qt = numpy.log( qt )
    # Rotations
    xx = 0.0; xy = 0.0; xz = 0.0; yy = 0.0; yz = 0.0; zz = 0.0
    mc = numpy.sum( mol.mass * mol.coor * mol.actv, axis = 0 ) / mt
    for i in range( size ):
        i3 = i * 3
        xx += mass[i] * ( mol.coor[sele[i],0] - mc[0] ) * ( mol.coor[sele[i],0] - mc[0] )
        xy += mass[i] * ( mol.coor[sele[i],0] - mc[0] ) * ( mol.coor[sele[i],1] - mc[1] )
        xz += mass[i] * ( mol.coor[sele[i],0] - mc[0] ) * ( mol.coor[sele[i],2] - mc[2] )
        yy += mass[i] * ( mol.coor[sele[i],1] - mc[1] ) * ( mol.coor[sele[i],1] - mc[1] )
        yz += mass[i] * ( mol.coor[sele[i],1] - mc[1] ) * ( mol.coor[sele[i],2] - mc[2] )
        zz += mass[i] * ( mol.coor[sele[i],2] - mc[2] ) * ( mol.coor[sele[i],2] - mc[2] )
    val, vec = numpy.linalg.eigh( numpy.array( [ yy+zz, -xy, -xz, -xy, xx+zz, -yz, -xz, -yz, xx+yy ] ).reshape( ( 3, 3 ) ) )
    t = ( 8.0 * numpy.pi * numpy.pi * qm3.data.KB * temp ) / ( qm3.data.H * qm3.data.H * qm3.data.NA ) * 1.0e-23
    qr = numpy.sqrt( numpy.pi * t * t * t * val[0] * val[1] * val[2] ) / symm
    qr = numpy.log( qr )
    # Vibrations
    t = 100.0 * qm3.data.C * qm3.data.H / ( qm3.data.KB * temp )
    qv = 1.0
    nf = 0
    for f in freq:
        if( f >= fcut ):
            qv /= ( 1.0 - numpy.exp( - f * t ) )
        else:
            nf += 1
    qv = numpy.log( qv )
    # ZPE && Gibbs (kJ/mol)
    # G = F + PV = - RT Ln Q + nRT   <<  (nRT cancels out when calculating relative terms...)
    zz = numpy.sum( freq[nf:] ) * 0.5 * 100.0 * qm3.data.C * qm3.data.H * qm3.data.NA * 1.0e-3
    gg = - qm3.data.R * temp * ( qt + qr + qv ) * 1.0e-3
    return( ( zz, gg ) )


