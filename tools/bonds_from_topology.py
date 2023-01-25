import	re
import	numpy
import	typing


def psf_bonds( fdsc: typing.IO ):
    bond = []
    l = fdsc.readline()
    while( l.find( "!NBOND" ) < 0 and l != "" ):
        l = fdsc.readline()
    nbnd = int( l.split()[0] )
    while( len( bond ) < nbnd ):
        tmp = [ int( i ) - 1 for i in fdsc.readline().split() ]
        for i in range( 0, len( tmp ), 2 ):
            bond.append( [ tmp[i], tmp[i+1] ] )
    return( bond )



def prmtop_bonds( fdsc: typing.IO ):
    __frmt = re.compile( "[aAiIeEdD]([0-9]+)" )
    hbnd = 0
    hlst = []
    xbnd = 0
    xlst = []
    l = fdsc.readline()
    while( l != "" ):
        if( l[0:14].upper() == "%FLAG POINTERS" ):
            l = fdsc.readline()
            dsp = int( __frmt.findall( l )[0] )
            l = fdsc.readline()
            hbnd = int( l[2*dsp:3*dsp] ) * 3
            xbnd = int( l[3*dsp:4*dsp] ) * 3
        elif( l[0:24].upper() == "%FLAG BONDS_INC_HYDROGEN" ):
            l = fdsc.readline()
            dsp = int( __frmt.findall( l )[0] )
            while( len( hlst ) < hbnd ):
                l = fdsc.readline()
                hlst += [ int( l[i:i+dsp] ) // 3 for i in range( 0, len( l ) - 1, dsp ) ]
        elif( l[0:28].upper() == "%FLAG BONDS_WITHOUT_HYDROGEN" ):
            l = fdsc.readline()
            dsp = int( __frmt.findall( l )[0] )
            while( len( xlst ) < xbnd ):
                l = fdsc.readline()
                xlst += [ int( l[i:i+dsp] ) // 3 for i in range( 0, len( l ) - 1, dsp ) ]
        l = fdsc.readline()
    bond = []
    for i in range( 0, xbnd, 3 ):
        bond.append( [ xlst[i], xlst[i+1] ] )
    for i in range( 0, hbnd, 3 ):
        bond.append( [ hlst[i], hlst[i+1] ] )
    bond.sort()
    return( bond )

