#!/usr/bin/env python3
import  numpy
import  qm3.data

def first_order_expansion( dene, rt ):
    """
 Chipot, C. Free Energy Calculations in Biological Systems. How Useful Are They in Practice?
        New Algorithms for Macromolecular Simulation, 2006, Springer) 3-540-25542-7
 Straatsma, T.P.; Berendsen, H.J.C.; Stam, A.J. Molecular Physics, 1986, 57, 89-95 [10.1080/00268978600100071]
 Smith, E.B.; Wells, B.H. Molecular Physics, 1984, 52, 701-704 [10.1080/00268978400101481]
 ----------------------------------------------------------------------------------------------
%DELTA F = - RT ln left langle e^{-{{U_j - U_i} over RT}} right rangle_i  +- RT {  { %delta %varepsilon } over { left langle e^{-{{U_j - U_i} over RT}} right rangle_i} }
~~~~~~~~
%delta %varepsilon^2 = { { 1 + 2 %tau } over N } left( left langle e^{-2{{U_j - U_i} over RT}} right rangle_i - left langle e^{-{{U_j - U_i} over RT}} right rangle_i^2 right)
~~~~~~~~
1 + 2 %tau = { 1 + r_1 } over { 1 - r_1 } 
~~~~~~~~
r_1 = { sum_{i=2}^N{left( x_i - bar x right) left( x_{i-1} - bar x right) } } over { sum_{i=1}^N{left( x_i - bar x right) ^2} }
    """
    exp = numpy.exp( - dene / rt )
    mm  = numpy.average( exp )
    tt  = exp - mm
    ex2 = numpy.average( numpy.exp( - 2.0 * dene / rt ) )
    r1  = sum( [ tt[i] * ( exp[i-1] - mm ) for i in range( 1, dene.shape[0] ) ] )
    r1 /= numpy.sum( numpy.square( tt ) )
    sr  = ( 1.0 + r1 ) / ( 1.0 - r1 )
    err = rt * numpy.sqrt( sr / dene.shape[0] * numpy.fabs( ex2 - mm * mm ) ) / mm
    return( { "n": dene.shape[0], "dF": - rt * numpy.log( mm ), "err": err, "sr": sr, "ac": r1 } )


rt = 300 * 1.e-3 * qm3.data.R
ac = 0.0
er = 0.0
print( "%6s%20s%20s%20s%20s"%( "Sampl.", "dF [kJ/mol]", "Error", "Samp. Ratio", "Auto.Corr." ) )
print( 86 * "-" )
for w in [ 1, .9, .8, .7, .6, .5, .4, .3, .2, .1 ]:
    with open( "cur_%.02lf"%( w ), "rt" ) as f:
        cen = numpy.array( [ float( i.strip() ) for i in f.readlines() ] )
    with open( "dsc_%.02lf"%( w ), "rt" ) as f:
        dsp = numpy.array( [ float( i.strip() ) for i in f.readlines() ] )
    res = first_order_expansion( dsp - cen, rt )
    ac += res["dF"]
    er += numpy.square( res["err"] )
    print( "%6d%20.10lf%20.10lf%20.10lf%20.10lf"%( res["n"], res["dF"], res["err"], res["sr"], res["ac"] ) )
print( 86 * "-" )
print( ac / 4.184, numpy.sqrt( er ) / 4.184, "_kcal/mol" )
