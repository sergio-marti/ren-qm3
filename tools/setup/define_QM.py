#!/usr/bin/env python3
import  numpy
import  qm3
import  qm3.utils._conn
import  pickle
import  re


def parse_prmtop( fname ):
    frmt = re.compile( "[aAiIeEdD]([0-9]+)" )
    hbnd = 0
    hlst = []
    xbnd = 0
    xlst = []
    with open( fname, "rt" ) as f:
        l = f.readline()
        while( l != "" ):
            if( l[0:14].upper() == "%FLAG POINTERS" ):
                l = f.readline()
                dsp = int( frmt.findall( l )[0] )
                l = f.readline()
                hbnd = int( l[2*dsp:3*dsp] ) * 3
                xbnd = int( l[3*dsp:4*dsp] ) * 3
            elif( l[0:24].upper() == "%FLAG BONDS_INC_HYDROGEN" ):
                l = f.readline()
                dsp = int( frmt.findall( l )[0] )
                while( len( hlst ) < hbnd ):
                    l = f.readline()
                    hlst += [ int( l[i:i+dsp] ) // 3 for i in range( 0, len( l ) - 1, dsp ) ]
            elif( l[0:28].upper() == "%FLAG BONDS_WITHOUT_HYDROGEN" ):
                l = f.readline()
                dsp = int( frmt.findall( l )[0] )
                while( len( xlst ) < xbnd ):
                    l = f.readline()
                    xlst += [ int( l[i:i+dsp] ) // 3 for i in range( 0, len( l ) - 1, dsp ) ]
            l = f.readline()
    bnd = []
    for i in range( 0, xbnd, 3 ):
        bnd.append( [ xlst[i], xlst[i+1] ] )
    for i in range( 0, hbnd, 3 ):
        bnd.append( [ hlst[i], hlst[i+1] ] )
    bnd.sort()
    return( bnd )


def parse_psf ( fname ):
    bnd = []
    with open( fname, "rt" ) as f:
        l = f.readline()
        while( l.upper().find( "!NBOND" ) < 0 ):
            l = f.readline()
        n = int( l.split()[0] )
        while( len( bnd ) < n ):
            t = [ int( i ) - 1 for i in f.readline().split() ]
            for i in range( 0, len( t ), 2 ):
                bnd.append( [ t[i], t[i+1] ] )
    return( bnd )


def get_atoms( con, ini, skp ):
    out = [ ini ]
    lst = 0
    while( len( out ) > lst ):
        lst = len( out )
        for i in out:
            for j in con[i]:
                if( not j in skp and not j in out ):
                    out.append( j )
    return( out )


if( __name__ == "__main__" ):
    m = qm3.molecule()
    m.prmtop_read( open( "complex.prmtop", "rt" ) )
    m.xyz_read( open( "nvt.xyz", "rt" ), replace = True )
    
    #b = parse_prmtop( "complex.prmtop" )
    #b = parse_psf( "complex.psf" )
    b = qm3.utils._conn.connectivity( 4, m.anum, m.coor, 0.12 )

    c = []
    for i in range( m.natm ):
        c.append( [] )
    for i,j in b:
        if( not j in c[i] ):
            c[i].append( j )
        if( not i in c[j] ):
            c[j].append( i )
    
    sqm = list( m.indx["A"][247].values() )
    sqm += get_atoms( c, m.indx["A"][63]["CB2"], [ m.indx["A"][62]["CA"], m.indx["A"][64]["C"], m.indx["A"][64]["CB"] ] )
    sqm.sort()
    print( sqm )
    
    with open( "sele_QM.pk", "wb" ) as f:
        pickle.dump( sqm, f )
    
    sla = []
    for i,j in b:
        if( i in sqm and not j in sqm ):
            sla.append( [ i, j ] )
        elif( j in sqm and not i in sqm ):
            sla.append( [ j, i ] )
    print( sla )
    
    with open( "sele_LA.pk", "wb" ) as f:
        pickle.dump( sla, f )
    
    tmp = numpy.zeros( m.natm, dtype=numpy.bool_ )
    tmp[sqm] = True
    sel = numpy.argwhere( m.sph_sel( tmp, 20 ) ).ravel().tolist()
    with open( "sele.pk", "wb" ) as f:
        pickle.dump( sel, f )
    
    smm = list( sorted( set( sel ).difference( set( sqm + sum( sla, [] ) ) ) ) )
    with open( "sele_MM.pk", "wb" ) as f:
        pickle.dump( smm, f )
    
    with open( "borra", "wt" ) as f:
        m.xyz_write( f, sele = tmp )
        tmp = numpy.zeros( m.natm, dtype=numpy.bool_ )
        tmp[[ j for i,j in sla]] = True
        m.xyz_write( f, sele = tmp )
        tmp = numpy.zeros( m.natm, dtype=numpy.bool_ )
        tmp[smm] = True
        m.xyz_write( f, sele = tmp )
