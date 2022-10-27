#!/usr/bin/env python3
import  sys
import  pickle

with open( sys.argv[2], "rb" ) as f:
    sqm = pickle.load( f )
sqm = [ i + 1 for i in sqm ]

flg = {}
bnd = 0
ang = 0
dih = 0
imp = 0
typ = 0
num = 0
t_rmin = {}
t_epsi = {}
rmin = []
epsi = []
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
                rmin.append( t_rmin[t[2]] )
                epsi.append( t_epsi[t[2]] )
                nat = int( t[0] )
                num = max( num, nat )
                if( nat in sqm ):
                    t[2] = str( typ )
                    t[3] = "0.0"
                    sys.stdout.write( "  ".join( t ) + " #[QM]\n" )
                else:
                    sys.stdout.write( l )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
            flg = { i: 0 for i in range( 1, num + 1 ) }
            for i in sqm:
                flg[i] = 1
        # ============================================================================
        elif( len( k ) > 0 and k[0].lower() == "masses" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 2 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 2 ):
                sys.stdout.write( l )
                typ += 1
                l = f.readline()
                t = l.split()
            typ += 1
            sys.stdout.write( " " + str( typ ) + " 1.0 #[QM]\n" )
            sys.stdout.write( l )
        # ============================================================================
        elif( len( k ) > 1 and k[0].lower() == "pair" and k[1].lower() == "coeffs" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 3 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 3 ):
                t_epsi[t[0]] = float( t[1] ) * 4.184
                t_rmin[t[0]] = float( t[2] ) * 0.5612310241546865
                sys.stdout.write( l )
                l = f.readline()
                t = l.split()
            sys.stdout.write( " " + str( typ ) + " 0.0 1.0 #[QM]\n" )
            sys.stdout.write( l )
        # ============================================================================
        elif( len( k ) > 1 and k[0].lower() == "bond" and k[1].lower() == "coeffs" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 3 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 3 ):
                sys.stdout.write( l )
                bnd += 1
                l = f.readline()
                t = l.split()
            bnd += 1
            sys.stdout.write( " " + str( bnd ) + " 0.0 0.0 #[QM]\n" )
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() == "bonds" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 4 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 4 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] ) < 2 ):
                    sys.stdout.write( l )
                else:
                    t[1] = str( bnd )
                    sys.stdout.write( "  " + "  ".join( t ) + " #[QM]\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        # ============================================================================
        elif( len( k ) > 1 and k[0].lower() == "angle" and k[1].lower() == "coeffs" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 3 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 3 ):
                sys.stdout.write( l )
                ang += 1
                l = f.readline()
                t = l.split()
            ang += 1
            sys.stdout.write( " " + str( ang ) + " 0.0 0.0 #[QM]\n" )
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() == "angles" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 5 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 5 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] + flg[int( t[4] )] ) < 3 ):
                    sys.stdout.write( l )
                else:
                    t[1] = str( ang )
                    sys.stdout.write( "  " + "  ".join( t ) + " #[QM]\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        # ============================================================================
        elif( len( k ) > 1 and k[0].lower() == "dihedral" and k[1].lower() == "coeffs" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 4 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 4 ):
                sys.stdout.write( l )
                dih += 1
                l = f.readline()
                t = l.split()
            dih += 1
            sys.stdout.write( " " + str( dih ) + " 0.0 0 0 0 #[QM]\n" )
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() == "dihedrals" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 6 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 5 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] + flg[int( t[4] )] + flg[int( t[5] )] ) < 4 ):
                    sys.stdout.write( l )
                else:
                    t[1] = str( dih )
                    sys.stdout.write( "  " + "  ".join( t ) + " #[QM]\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        # ============================================================================
        elif( len( k ) > 1 and k[0].lower() == "improper" and k[1].lower() == "coeffs" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 3 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 3 ):
                sys.stdout.write( l )
                imp += 1
                l = f.readline()
                t = l.split()
            imp += 1
            sys.stdout.write( " " + str( imp ) + " 0.0 180 #[QM]\n" )
            sys.stdout.write( l )
        elif( len( k ) > 0 and k[0].lower() == "impropers" ):
            sys.stdout.write( l )
            l = f.readline()
            while( len( l.split() ) < 6 ):
                sys.stdout.write( l )
                l = f.readline()
            t = l.split()
            while( len( t ) >= 5 ):
                if( ( flg[int( t[2] )] + flg[int( t[3] )] + flg[int( t[4] )] + flg[int( t[5] )] ) < 4 ):
                    sys.stdout.write( l )
                else:
                    t[1] = str( imp )
                    sys.stdout.write( "  " + "  ".join( t ) + " #[QM]\n" )
                l = f.readline()
                t = l.split()
            sys.stdout.write( l )
        # ============================================================================
        else:
            sys.stdout.write( l )
        l = f.readline()

print( "\n# " + 80 * "-" )
print( "# Ttypes:    ", str( typ ) )
print( "# Tbonds:    ", str( bnd ) )
print( "# Tangles:   ", str( ang ) )
print( "# Tdihedrals:", str( dih ) )
print( "# Timpropers:", str( imp ) )
print( "# " + 80 * "-" )

with open( "lj_parm.pk", "wb" ) as f:
    pickle.dump( rmin, f )
    pickle.dump( epsi, f )
