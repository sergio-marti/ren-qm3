import  numpy

def exclusions( natm, bond, sele ):
    conn = []
    for i in range( natm ):
        conn.append( [] )
    for i,j in bond:
        conn[i].append( j )
        conn[j].append( i )
    excl = []
    nx12 = 0
    nx13 = 0
    nx14 = 0
    for i in numpy.argwhere( sele.ravel() ).ravel():
        for j in conn[i]:
            if( j != i and not sele[j] ):
                excl.append( [ i, j, 0.0 ] )
                nx12 += 1
            for k in conn[j]:
                if( k != i and not sele[k] ):
                    excl.append( [ i, k, 0.0 ] )
                    nx13 += 1
                for l in conn[k]:
                    if( k != i and l != j and l != i and not sele[l] ):
                        excl.append( [ i, l, 0.5 ] )
                        nx14 += 1
    print( ">> %d exclusions generated (1-2:%d, 1-3:%d, 1-4:%d)"%( nx12 + nx13 + nx14, nx12, nx13, nx14 ) )
    return( excl )
