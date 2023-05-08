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
        self.exe_ene = "ridft  1>  tmole.log 2>> tmole.log"
        self.exe_grd = "rdgrad 1>> tmole.log 2>> tmole.log"
#        self.exe_grd = "ricc2  1>> tmole.log 2>> tmole.log"


    def mk_input( self, mol, run ):
        dq = numpy.zeros( mol.natm )
        f = open( "coord", "wt" )
        f.write( "$coord\n" )
        for i in self.sel:
            f.write( "%20.10lf%20.10lf%20.10lf%4s\n"%(
                mol.coor[i,0] * self.cx, mol.coor[i,1] * self.cx, mol.coor[i,2] * self.cx,
                qm3.data.symbol[mol.anum[i]] ) )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
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
                f.write( "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] + dq[i] ) )
            f.write( "$end" )
            f.close()


    def parse_log( self, mol, run ):
#        with open( "energy", "rt" ) as f:
#            f.readline()
#            out = float( f.readline().split()[1] ) * self.ce
#            mol.func += out
        os.unlink( "energy" )
        if( run == "grad" ):
            f = open( "gradient", "rt" )
            f.readline()
            out = float( f.readline().split()[-4] ) * self.ce
            mol.func += out
#            for i in range( 2 + len( self.sel ) + len( self.lnk ) ):
            for i in range( len( self.sel ) + len( self.lnk ) ):
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
        return( out )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe_ene )
        os.system( self.exe_grd )
        return( self.parse_log( mol, "grad" ) )






ri_dft = """
define << EOD

slave
a coord
*
no
b all def2-TZVP
*
eht
y
[MOLECULAR_CHARGE]
y
ri
on
*
dft
func wb97x-d
on
*
*
EOD
"""

ri_mp2 = """
define << EOD

slave
a coord
*
no
b all def2-TZVP
*
eht
y
[MOLECULAR_CHARGE]
y
ri
on
*
cc
cbas
*
ricc2
mp2
geoopt mp2
*
*
*
EOD
"""

fix_qmmm = """
# >> modify "control" file for QM/MM calculations:
$scfiterlimit 200

$drvopt         
   point charges

$point_charges file=charges
$point_charge_gradients file=charges.gradient
"""

fix_smp = """
# >> modify "control" file for SMP calculations:

$smp_cpus 64
"""


ri_cc2_es = """
define << EOD

slave
a coord
*
no
b all def2-TZVP
*
eht
y
[MOLECULAR_CHARGE]
y
ri
on
*
cc
cbas
*
ricc2
cc2
geoopt cc2 (a 1)
*
exci
irrep=a nexc=5
spectrum states=all operators=diplen,qudlen,angmom,dipvel
*
*
*
EOD
"""

