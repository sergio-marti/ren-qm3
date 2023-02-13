import  numpy
import  typing
import  os
import  qm3.data
import  qm3.utils
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.cx  = 1.0 / qm3.data.A0
        self.ce  = qm3.data.H2J
        self.cg  = self.ce * self.cx
        self.exe_ene = "dscf 1>  tmole.log 2>> tmole.log"
        self.exe_grd = "grad 1>> tmole.log 2>> tmole.log"


    def mk_input( self, mol, run ):
        dq = numpy.zeros( mol.natm )
        f = open( "coord", "wt" )
        f.write( "$coord\n" )
        for i in self.sel:
            f.write( "%20.10lf%20.10lf%20.10lf%4s\n"%(
                mol.coor[i,0] * self.cx, mol.coor[i,1] * self.cx, mol.coor[i,2] * self.cx,
                qm3.data.symbol[mol.anum[i]] )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i,j in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                f.write( "%20.10lf%20.10lf%20.10lf   H\n"%( c[0] * self.cx, c[1] * self.cx, c[2] * self.cx ) )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
            # redistribute MM-charge on the remaining atoms of the group
            for i,j in self.lnk:
                if( j in self.grp ):
                    dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
            # ----------------------------------------------------------
        f.write( "$user-defined bonds\n$end" )
        f.close()
        if( len( self.nbn ) > 0 ):
            f = open( "charges", "wt" )
            f.write( "$point_charges nocheck\n" )
            for i in self.nbn:
                tmp = ( mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 ) ) * self.cx
                f.write( "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] + dq[i] )
            f.write( "$end" )
            f.close()


    def parse_log( self, mol, run ):
        with open( "energy", "rt" ) as f:
            f.readline()
            mol.func += float( f.readline().split()[1] ) * self.ce
        os.unlink( "energy" )
        if( run == "grad" ):
            f = open( "gradient", "rt" )
            for i in range( 2 + len( self.sel ) + len( self.lnk ) ):
                f.readline()
            g = []
            for i in range( len( self.sel ) + len( self.lnk ) ):
                g += [ float( j.replace( "D", "E" ) ) * self.cg for j in f.readline().strip().split() ]
            qm3.engines.Link_grad( self.vla, g )
            k = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[k]
                    k += 1
            f.close()
            os.unlink( "gradient" )
            if( len( self.nbn ) > 0 ):
                f = open( "charges.gradient", "rt" )
                f.readline()
                for i in self.nbn:
                    t = [ float( j.replace( "D", "E" ) ) * self.cg for j in f.readline().strip().split() ]
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += t[j]
                f.close()
                os.unlink( "charges.gradient" )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe_ene )
        os.system( self.exe_grd )
        self.parse_log( mol, "grad" )






"""
# >> system setup (previous to any calculation)

x2t xyz > coord
define << EOD

slave
a coord
*
no
b all def2-TZVP      << BASIS SET
*
eht
y
0                   << MOLECULAR CHARGE
y
dft
func wb97x-d        << FUNCTIONAL
on
*
*
EOD



# >> modify "control" file for QM/MM calculations

$drvopt
   point charges

$point_charges file=charges
$point_charge_gradients file=charges.gradient
$end
"""
