import  numpy
import  typing
import  collections
import  re
import  qm3.data
import  qm3.utils


class molecule( object ):
    """
        selections are based on numpy.bool arrays
        use numpy.logical_[and/or/not] to perform complex selections

        apply numpy.argwhere( ).ravel() to obtain the indices for the engines
    """
    def __init__( self ):
        self.natm = 0
        self.boxl = numpy.array( [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] )
        self.labl = None # dtype=numpy.unicode
        self.coor = None # dtype=numpy.float64
        self.segn = None # dtype=numpy.unicode
        self.resi = None # dtype=numpy.int32
        self.resn = None # dtype=numpy.unicode
        self.anum = None # dtype=numpy.int16
        self.chrg = None # dtype=numpy.float32
        self.mass = None # dtype=numpy.float32
        self.rlim = None # dtype=numpy.int32
        self.actv = None # dtype=numpy.bool
        self.indx = None
        self.engines = []
        self.func = 0.0
        self.grad = None # dtype=numpy.float64
        self.rval = None


    def current_step( self, step ):
        pass


    def rebuild( self ):
        # -- resiude boundaries
        rlim = []
        l_seg = None
        l_rsn = None
        l_rsi = None
        for i in range( self.natm ):
            if( l_seg != self.segn[i] or l_rsn != self.resn[i] or l_rsi != self.resi[i] ):
                rlim.append( i )
                l_seg = self.segn[i]
                l_rsi = self.resi[i]
                l_rsn = self.resn[i]
        rlim.append( self.natm )
        self.rlim = numpy.array( rlim, dtype=numpy.int32 )
        # -- atom indexing
        self.indx = collections.OrderedDict()
        for s in numpy.unique( self.segn ):
            self.indx[s] = collections.OrderedDict()
        for i in range( len( self.rlim ) - 1 ):
            self.indx[self.segn[self.rlim[i]]][self.resi[self.rlim[i]]] = collections.OrderedDict()
            for j in range( self.rlim[i], self.rlim[i+1] ):
                self.indx[self.segn[self.rlim[i]]][self.resi[self.rlim[i]]][self.labl[j]] = j



    def sph_sel( self, sele: numpy.array, radius: float ) -> numpy.array:
        out = numpy.zeros( self.natm, dtype=numpy.bool )
        siz = sele.sum()
        idx = numpy.argwhere( sele ).ravel()
        cen = numpy.sum( self.coor[sele], axis=0 ) / siz
        dsp = max( map( lambda c: qm3.utils.distanceSQ( cen, c ), self.coor[sele] ) )
        cut = numpy.power( radius + numpy.sqrt( dsp ) + 0.1, 2.0 )
        rad = radius * radius
        res = []
        for k0 in range( len( self.rlim ) - 1 ):
            k1 = self.rlim[k0]
            kn = self.rlim[k0+1]
            kf = False
            while( k1 < kn and not kf ):
                kf |= qm3.utils.distanceSQ( cen, self.coor[k1], self.boxl ) <= cut
                k1 += 1
            if( kf ):
                k1 = self.rlim[k0]
                kf = False
                while( k1 < kn and not kf ):
                    i1 = 0
                    while( i1 < siz and not kf ):
                        kf |= qm3.utils.distanceSQ( self.coor[k1], self.coor[idx[i1]], self.boxl ) <= rad
                        i1 += 1
                    k1 += 1
                if( kf ):
                    out[self.rlim[k0]:kn] = True
        return( out )


    def copy( self,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ) ):
        if( sele.sum() > 0 ):
            lsel = sele
        else:
            lsel = numpy.ones( self.natm, dtype=numpy.bool )
        out = molecule()
        out.natm = lsel.sum()
        out.boxl = numpy.array( [ qm3.data.MXLAT, qm3.data.MXLAT, qm3.data.MXLAT ] )
        out.labl = self.labl[lsel]
        out.coor = self.coor[lsel]
        out.segn = self.segn[lsel]
        out.resi = self.resi[lsel]
        out.resn = self.resn[lsel]
        out.anum = self.anum[lsel]
        out.chrg = self.chrg[lsel]
        out.mass = self.mass[lsel]
        out.actv = numpy.ones( ( out.natm, 1 ), dtype=numpy.bool )
        out.rebuild()
        return( out )


    def guess_atomic_numbers( self ):
        if( self.mass.sum() > 0 ):
            anum = []
            for i in range( self.natm ):
                anum.append( int( numpy.where( numpy.fabs( qm3.data.mass - self.mass[i] ) < 0.2 )[0][0] ) )
            self.anum = numpy.array( anum, dtype=numpy.int16 )
        else:
            mass = []
            anum = []
            for i in range( self.natm ):
                anum.append( qm3.data.rsymbol["".join( [ j for j in self.labl[i] if j.isalpha() ] ).title()] )
                mass.append( qm3.data.mass[anum[i]] )
            self.anum = numpy.array( anum, dtype=numpy.int16 )
            self.mass = numpy.array( mass, dtype=numpy.float32 )


    def fill_masses( self ):
        if( self.anum.sum() > 0 ):
            self.mass = qm3.data.mass[self.anum]
        else:
            self.guess_atomic_numbers()


    def set_active( self,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ) ):
        if( sele.sum() > 0 ):
            self.actv = sele.reshape( ( self.natm, 1 ) )
        else:
            self.actv = numpy.ones( ( self.natm, 1 ), dtype=numpy.bool )

# =================================================================================================

    def pdb_read( self, fdsc ):
        """
          1         2         3         4         5         6         7         8
.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.
ATOM     31 HG22 ILE     3      -8.509  29.691  -4.228  0.00  0.00      A
ATOM    771  H1  TIP3  253       6.588   9.359  -8.787  1.00  0.00      B    H
ATOM  00000  OH2 HOH  3314     -11.039 -22.605  29.142  0.00  0.00      B4
ATOM  05681  SOD SOD   146     -37.840 -41.531   8.396  0.00  0.00      ION1

          1         2         3         4         5         6         7         8
.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.
HETATM   84  OH2 HOH A  28      52.537   5.370  44.344                          
HETATM   85  H1  HOH A  29       8.127  45.914  57.300                          
HETATM   86  H2  HOH A  29       9.503  46.512  57.945                          

          1         2         3         4         5         6         7         8
.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.
ATOM   7921  O   WAT  2632     -12.409 -10.338 -10.063  1.00  0.00
ATOM   7922  H1  WAT  2632     -11.616 -10.833 -10.270  1.00  0.00
ATOM   7923  H2  WAT  2632     -12.115  -9.659  -9.455  1.00  0.00
        """
        self.natm = 0
        labl = []
        coor = []
        segn = []
        resi = []
        resn = []
        for l in fdsc:
            if( l[0:4] == "ATOM" or l[0:4] == "HETA" ):
                labl.append( l[12:17].strip() )
                if( l[21] != " " ):
                    resn.append( l[17:21].strip() )
                    resi.append( int( l[22:26] ) )
                    segn.append( l[21] )
                else:
                    resn.append( l[17:22].strip() )
                    resi.append( int( l[22:26] ) )
                    if( len( l ) > 70 ):
                        segn.append( l[72:].strip().split()[0] )
                    else:
                        segn.append( "A" )
                coor += [ float( l[30:38] ), float( l[38:46] ), float( l[46:54] ) ]
                self.natm += 1
        self.labl = numpy.array( labl, dtype=numpy.unicode )
        self.coor = numpy.array( coor, dtype=numpy.float64 ).reshape( ( self.natm, 3 ) )
        self.segn = numpy.array( segn, dtype=numpy.unicode )
        self.resi = numpy.array( resi, dtype=numpy.int32 )
        self.resn = numpy.array( resn, dtype=numpy.unicode )
        self.anum = numpy.zeros( self.natm, dtype=numpy.int16 )
        self.chrg = numpy.zeros( self.natm, dtype=numpy.float32 )
        self.mass = numpy.zeros( self.natm, dtype=numpy.float32 )
        self.actv = numpy.ones( ( self.natm, 1 ), dtype=numpy.bool )
        self.engines = []
        self.func = 0.0
        self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        self.rebuild()


    def pdb_write( self, fdsc,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ) ):
        if( sele.sum() > 0 ):
            lsel = sele
        else:
            lsel = numpy.ones( self.natm, dtype=numpy.bool )
        fdsc.write( "REMARK %12.4le %12.4le %12.4le\n"%( self.boxl[0], self.boxl[1], self.boxl[2] ) )
        j = 0
        for i in range( self.natm ):
            if( lsel[i] ):
                fdsc.write( "ATOM  %5d %-5s%-5s%4d    %8.3lf%8.3lf%8.3lf%6.2lf%6.2lf      %-4s\n"%( ( j % 99999 ) + 1, 
                    " " * ( len( self.labl[i] ) < 4 ) + self.labl[i],
                    self.resn[i], self.resi[i], self.coor[i,0], self.coor[i,1], self.coor[i,2], 
                    0.0, 0.0, self.segn[i] ) )
                j += 1
        fdsc.write( "END\n" )

# =================================================================================================

    def xyz_read( self, fdsc,
            replace: typing.Optional[bool] = False ):
        n = int( fdsc.readline().strip() )
        fdsc.readline()
        if( replace and self.natm == n ):
            for i in range( n ):
                temp = f.readline().split()
                for j in [0, 1, 2]:
                    self.coor[i,j] = float( temp[j+1] )
        else:
            self.natm = 0
            labl = []
            coor = []
            labl = []
            for i in range( n ):
                temp = fdsc.readline().split()
                labl.append( temp[0] )
                coor += [ float( temp[1] ), float( temp[2] ), float( temp[3] ) ]
                self.natm += 1
            self.labl = numpy.array( labl, dtype=numpy.unicode )
            self.coor = numpy.array( coor, dtype=numpy.float64 ).reshape( ( self.natm, 3 ) )
            temp = [ "X" ] * self.natm
            self.segn = numpy.array( temp, dtype=numpy.unicode )
            self.resi = numpy.ones( self.natm, dtype=numpy.int16 )
            self.resn = numpy.array( temp, dtype=numpy.unicode )
            self.anum = numpy.zeros( self.natm, dtype=numpy.int16 )
            self.chrg = numpy.zeros( self.natm, dtype=numpy.float32 )
            self.mass = numpy.zeros( self.natm, dtype=numpy.float32 )
            self.actv = numpy.ones( ( self.natm, 1 ), dtype=numpy.bool )
            self.rlim = numpy.array( [ 0, self.natm ], dtype=numpy.int32 )
            self.indx = None
            self.engines = []
            self.func = 0.0
            self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )


    def xyz_write( self, fdsc,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            formt: typing.Optional[str] = "%20.10lf" ):
        fmt = "%-4s" + 3 * formt + "\n"
        if( sele.sum() > 0 ):
            lsel = sele
        else:
            lsel = numpy.ones( self.natm, dtype=numpy.bool )
        siz = lsel.sum()
        fdsc.write( "%d\n"%( siz ) )
        fdsc.write( "%12.4le %12.4le %12.4le\n"%( self.boxl[0], self.boxl[1], self.boxl[2] ) )
        for i in range( self.natm ):
            if( lsel[i] ):
                fdsc.write( fmt%( qm3.data.symbol[self.anum[i]], self.coor[i,0], self.coor[i,1], self.coor[i,2] ) )

# =================================================================================================

    def psf_read( self, fdsc ):
        if( fdsc.readline().split()[0] == "PSF" ):
            fdsc.readline()
            for i in range( int( fdsc.readline().split()[0] ) + 1 ):
                fdsc.readline()
            natm = int( fdsc.readline().split()[0] )
            flag = True
            if( self.natm == natm ):
                chrg = []
                mass = []
                for i in range( self.natm ):
                    temp = fdsc.readline().split()
                    if( self.segn[i] == temp[1] and self.resi[i] == int( temp[2] ) and self.resn[i] == temp[3] and self.labl[i] == temp[4]  ):
                        chrg.append( float( temp[6] ) )
                        mass.append( float( temp[7] ) )
                    else:
                        print( "- Wrong PSF data (%d): %s/%s %d/%s %s/%s %s/%s"%( i+1, self.segn[i], temp[1],
                            self.resi[i], temp[2], self.resn[i], temp[3], self.labl[i], temp[4] ) )
                        flag = False
            else:
                print( "- Invalid number of atoms in PSF!" )
                flag = False
            if( flag ):
                self.chrg = numpy.array( chrg, dtype=numpy.float32 )
                self.mass = numpy.array( mass, dtype=numpy.float32 )


    def prmtop_read( self, fdesc ):
        __frmt = re.compile( "[aAiIeEdD]([0-9]+)" )
        l = fdesc.readline()
        while( l != "" ):
            if( l[0:12].upper() == "%FLAG CHARGE" ):
                dsp = int( __frmt.findall( fdesc.readline() )[0] )
                chrg = []
                while( len( chrg ) < self.natm ):
                    l = fdesc.readline()
                    chrg += [ float( l[i:i+dsp] ) / 18.2223 for i in range( 0, len( l ) - 1, dsp ) ]
                self.chrg = numpy.array( chrg, dtype=numpy.float32 )
            elif( l[0:19].upper() == "%FLAG ATOMIC_NUMBER" ):
                anum = []
                l = fdesc.readline()
                dsp = int( __frmt.findall( l )[0] )
                while( len( anum ) < self.natm ):
                    l = fdesc.readline()
                    anum += [ int( l[i:i+dsp] ) for i in range( 0, len( l ) - 1, dsp ) ]
                self.anum = numpy.array( anum, dtype=numpy.int16 )
            elif( l[0:10].upper() == "%FLAG MASS" ):
                mass = []
                l = fdesc.readline()
                dsp = int( __frmt.findall( l )[0] )
                while( len( mass ) < self.natm ):
                    l = fdesc.readline()
                    mass += [ float( l[i:i+dsp] ) for i in range( 0, len( l ) - 1, dsp ) ]
                self.mass = numpy.array( mass, dtype=numpy.float32 )
            l = fdesc.readline()

# =================================================================================================

# dcd_read / dcd_next; dcd_write / dcd_append

# =================================================================================================

    def get_func( self ):
        self.rval = []
        self.func = 0.0
        self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        for itm in self.engines:
            itm.get_func( self )


    def get_grad( self ):
        self.rval = []
        self.func = 0.0
        self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        for itm in self.engines:
            itm.get_grad( self )
        self.grad *= self.actv
