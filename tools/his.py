#!/usr/bin/env python3
import  sys
import  numpy
import  qm3
import  qm3.utils


m = qm3.molecule()
m.pdb_read( open( sys.argv[1] ) )

his = {}
for i in range( m.natm ):
    if( m.resn[i] == "HIS" and m.labl[i] == "ND1" ):
        if( not m.segn[i] in his ):
            his[m.segn[i]] = {}
        if( not m.resi[i] in his[m.segn[i]] ):
            his[m.segn[i]][m.resi[i]] = {}
        if( not "ND1" in his[m.segn[i]][m.resi[i]] ):
            his[m.segn[i]][m.resi[i]]["ND1"] = i
    elif( m.resn[i] == "HIS" and m.labl[i] == "NE2" ):
        if( not m.segn[i] in his ):
            his[m.segn[i]] = {}
        if( not m.resi[i] in his[m.segn[i]] ):
            his[m.segn[i]][m.resi[i]] = {}
        if( not "NE2" in his[m.segn[i]][m.resi[i]] ):
            his[m.segn[i]][m.resi[i]]["NE2"] = i

cut = 3.2
for chain in sorted( his ):
    for resi in sorted( his[chain] ):
        jd = m.indx[chain][resi]["ND1"]
        je = m.indx[chain][resi]["NE2"]
        print( chain, resi )
        t  = numpy.zeros( m.natm, dtype=numpy.bool_ )
        t[his[chain][resi]["ND1"]] = True
        sd = numpy.argwhere( m.sph_sel( t, 4. ) ).ravel()
        t  = numpy.zeros( m.natm, dtype=numpy.bool_ )
        t[his[chain][resi]["NE2"]] = True
        se = numpy.argwhere( m.sph_sel( t, 4. ) ).ravel()
        print( "\t- HSD:" )
        for i in sd:
            if( m.labl[i] in [ "O", "SD" ] ):
                t = round( qm3.utils.distance( m.coor[jd], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[Hd] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
            elif( m.labl[i][0:2] in [ "OE", "OD" ] ):
                t = round( qm3.utils.distance( m.coor[jd], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[Hd] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
        for i in se:
            if( m.labl[i] in [ "N", "SG", "OG", "NZ", "NE", "NH1", "NH2", "ND2", "NE1" ] ):
                t = round( qm3.utils.distance( m.coor[je], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[ e] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
            elif( m.labl[i] == "NE2" and m.resn[i] == "TRP" ):
                t = round( qm3.utils.distance( m.coor[je], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[ e] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
        print( "\n\t- HSE:" )
        for i in sd:
            if( m.labl[i] in [ "N", "SG", "OG", "NZ", "NE", "NH1", "NH2", "ND2", "NE1" ] ):
                t = round( qm3.utils.distance( m.coor[jd], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[ d] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
            elif( m.labl[i] == "NE2" and m.resn[i] == "TRP" ):
                t = round( qm3.utils.distance( m.coor[jd], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[ d] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
        for i in se:
            if( m.labl[i] in [ "O", "SD" ] ):
                t = round( qm3.utils.distance( m.coor[je], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[He] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
            elif( m.labl[i][0:2] in [ "OE", "OD" ] ):
                t = round( qm3.utils.distance( m.coor[je], m.coor[i] ), 3 )
                if( t <= cut  ):
                    print( "\t\t[He] %4s %4d %4s %4s"%( m.segn[i], m.resi[i], m.resn[i], m.labl[i] ), t )
        print( 80*"#" )
