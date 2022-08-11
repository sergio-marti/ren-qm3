#!/usr/bin/env python3
import  sys
import  pickle

with open( sys.argv[2], "rb" ) as f:
    sqm = pickle.load( f )
sqm = [ i + 1 for i in sqm ]

print( "# " + 80 * "-" )
print( "# group qmatm id " + " ".join( [ str( i ) for i in sqm ] ) )
print( "# neigh_modify exclude group qmatm qmatm" )
print( "# " + 80 * "-" )

flg = {}
with open( sys.argv[1], "rt" ) as f:
    l = f.readline()
    while( l != "" ):
        k = l.split()
        if( len( k ) > 0 and k[0].lower() == "atoms" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 7 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 7 ):
                nat = int( t[0] )
                if( nat in sqm ):
                    t[3] = "0.0"
                    sys.stdout.write( "  ".join( t ) + "\n" )
                else:
                    sys.stdout.write( l )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
            flg = { i: 0 for i in range( 1, nat + 1 ) }
            for i in sqm:
                flg[i] = 1
        elif( len( k ) > 0 and k[0].lower() == "bonds" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 4 ):
                sys.stdout.write( l )
                l = f.readline()
            c = 0
            t = l.split()
            while( len( t ) >= 4 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] ) < 2 ):
                    c += 1
                    t[0] = str( c )
                    sys.stdout.write( "  " + "  ".join( t ) + "\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() == "angles" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 5 ):
                sys.stdout.write( l )
                l = f.readline()
            c = 0
            t = l.split()
            while( len( t ) >= 5 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] + flg[int( t[4] )] ) < 3 ):
                    c += 1
                    t[0] = str( c )
                    sys.stdout.write( "  " + "  ".join( t ) + "\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() in [ "dihedrals", "impropers" ] ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 6 ):
                sys.stdout.write( l )
                l = f.readline()
            c = 0
            t = l.split()
            while( len( t ) >= 5 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] + flg[int( t[4] )] + flg[int( t[5] )] ) < 4 ):
                    c += 1
                    t[0] = str( c )
                    sys.stdout.write( "  " + "  ".join( t ) + "\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        else:
            sys.stdout.write( l )
        l = f.readline()
