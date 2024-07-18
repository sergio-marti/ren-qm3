import  numpy
import  typing
import  math
import  qm3.data


#
# J. Comp. Phys. v22, p245 (1976) [10.1016/0021-9991(76)90078-4]
# Phys. Rev. Lett. v91, p140601 (2003) [10.1103/PhysRevLett.91.140601]
# J. Phys. Chem. B v114, p10235 (2010) [10.1021/jp102971x]
# https://github.com/choderalab/pymbar.git
#
def bennett_acceptance_ratio( forward: numpy.array,
                reverse: numpy.array,
                temperature: typing.Optional[float] = 300. ) -> dict:

    def __diff( size, exp_f, exp_r, dF ):
        edf = math.exp( - dF )
        f_f = 0.0
        f_r = 0.0
        for i in range( size ):
            f_f += 1.0 / ( 1.0 + edf / exp_f[i] )
            f_r += 1.0 / ( 1.0 + 1.0 / ( exp_r[i] * edf ) )
        return( math.log( f_f ) - math.log( f_r ) )

    rt = 1.0e-3 * qm3.data.R * temperature
    size  = min( forward.shape[0], reverse.shape[0] )
    exp_f = numpy.exp( - forward[0:size] / rt )
    exp_r = numpy.exp( - reverse[0:size] / rt )
    x0 = - math.log( numpy.mean( exp_f ) )
    xf =   math.log( numpy.mean( exp_r ) )
    f0 = __diff( size, exp_f, exp_r, x0 )
    ff = __diff( size, exp_f, exp_r, xf )
    while( f0 * ff > 0.0 ):
        x0 -= 0.1
        f0 = __diff( size, exp_f, exp_r, x0 )
        xf += 0.1
        ff = __diff( size, exp_f, exp_r, xf )
    fm = 1.0
    i  = 0
    ii = 1000
    while( i < ii and math.fabs( fm ) > 1.e-8 ):
        xm = ( x0 + xf ) * 0.5
        fm = __diff( size, exp_f, exp_r, xm )
        if( f0 * fm <= 0.0 ):
            xf = xm
            ff = fm
        elif( ff * fm <= 0.0 ):
            x0 = xm
            f0 = fm
        else:
            return( None )
        i += 1
    if( i >= ii ):
        return( None )
    edf = math.exp( - xm )
    af  = 0.0
    af2 = 0.0
    ar  = 0.0
    ar2 = 0.0
    for i in range( size ):
        t    = 1.0 / ( 1.0 + edf / exp_f[i] )
        af  += t
        af2 += t * t
        t    = 1.0 / ( 1.0 + 1.0 / ( exp_r[i] * edf ) )
        ar  += t
        ar2 += t * t
    af  = af  / size
    ar  = ar  / size
    af2 = af2 / size
    ar2 = ar2 / size
    return( { "dF": rt * xm, "Error": rt * math.sqrt( ( af2 / ( af * af ) + ar2 / ( ar * ar ) - 2.0 ) / size ) } )



#
# First order expansion
# ----------------------------------------------------------------------------------------------
# Chipot, C. Free Energy Calculations in Biological Systems. How Useful Are They in Practice?
#        New Algorithms for Macromolecular Simulation, 2006, Springer (3-540-25542-7)
# Straatsma, T.P.; Berendsen, H.J.C.; Stam, A.J. Molecular Physics, 1986, 57, 89-95 [10.1080/00268978600100071]
# Smith, E.B.; Wells, B.H. Molecular Physics, 1984, 52, 701-704 [10.1080/00268978400101481]
# ----------------------------------------------------------------------------------------------
#%DELTA F = - RT ln left langle e^{-{{U_j - U_i} over RT}} right rangle_i  +- RT {  { %delta %varepsilon } over { left langle e^{-{{U_j - U_i} over RT}} right rangle_i} }
#~~~~~~~~
#%delta %varepsilon^2 = { { 1 + 2 %tau } over N } left( left langle e^{-2{{U_j - U_i} over RT}} right rangle_i - left langle e^{-{{U_j - U_i} over RT}} right rangle_i^2 right)
#~~~~~~~~
#1 + 2 %tau = { 1 + r_1 } over { 1 - r_1 } 
#~~~~~~~~
#r_1 = { sum_{i=2}^N{left( x_i - bar x right) left( x_{i-1} - bar x right) } } over { sum_{i=1}^N{left( x_i - bar x right) ^2} }
# ----------------------------------------------------------------------------------------------
#
def fep_integrate( dene: numpy.array, temperature: typing.Optional[float] = 300.0 ) -> dict:
    rt  = temperature * 1.0e-3 * qm3.data.R
    exp = numpy.exp( - dene / rt )
    mm  = numpy.mean( exp )
    ex2 = numpy.mean( numpy.exp( - 2.0 * dene / rt ) )
    nn  = dene.shape[0]
    try:
        r1  = sum( [ ( exp[i] - mm ) * ( exp[i-1] - mm ) for i in range( 1, nn ) ] )
        r1 /= sum( [ math.pow( exp[i] - mm, 2.0 ) for i in range( nn ) ] )
    except:
        r1 = 0.0
    sr  = ( 1.0 + r1 ) / ( 1.0 - r1 )
    err = rt * math.sqrt( sr / float( nn ) * math.fabs( ex2 - mm * mm ) ) / mm
    out = { "Samples": nn, "dF": - rt * math.log( mm ), "Error": err, "Sampling Ratio": sr, "Autocorrelation": r1 }
    return( out )
