import  numpy
import  typing
import  os
import  re
import  glob
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, fdsc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.exe = "bash r.orca"
        self.inp = fdsc.read()
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0


    def mk_input( self, mol, run ):
        s_qm = ""
        for i in self.sel:
            s_qm += "%2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = qm3.engines.Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                s_qm += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        s_mm = ""
        if( len( self.nbn ) > 0 ):
            s_mm = "%pointcharges \"orca.pc\""
            f = open( "orca.pc", "wt" )
            f.write( "%d\n"%( len( self.nbn ) ) )
            for i in self.nbn:
                tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
                f.write( "%12.4lf%20.10lf%20.10lf%20.10lf\n"%( mol.chrg[i], tmp[0], tmp[1], tmp[2] ) )
            f.close()
        s_rn = ""
        if( run == "grad" ):
            s_rn = "engrad"
        f = open( "orca.inp", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_charges", s_mm )
        f.write( buf )
        f.close()


    def parse_log( self, mol, run ):
        if( run == "grad" ):
            f = open( "orca.engrad", "rt" )
            t = re.compile( "[0-9\.\-]+" ).findall( f.read() )
            f.close()
            n = int( t[0] )
            mol.func += float( t[1] ) * self.ce
            g = [ float( t[i] ) * self.cg for i in range( 2, 2 + n * 3 ) ]
            qm3.engines.Link_grad( self.vla, g )
            k = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[k]
                    k += 1
            if( len( self.nbn ) > 0 and os.access( "orca.pcgrad", os.R_OK ) ):
                f = open( "orca.pcgrad", "rt" )
                t = re.compile( "[0-9\.\-]+" ).findall( f.read() )
                f.close()
                n = int( t[0] )
                g = [ float( t[i] ) * self.cg for i in range( 1, 1 + n * 3 ) ]
                k = 0
                for i in self.nbn:
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += g[k]
                        k += 1
        else:
            f = open( "orca.out", "rt" )
            mol.func += self.ce * float( re.compile( "FINAL SINGLE POINT ENERGY[\ ]*([0-9\.\-]+)" ).findall( f.read() )[0] )
            f.close()
        # parse orca output in search for "^CHELPG Charges" (only if chelpg is found in self.inp)
        for ff in glob.glob( "orca.*" ):
            if( ff != "orca.gbw" and ff != "orca.ges" ):
                os.unlink( ff )


    def get_func( self, mol ):
        self.mk_input( mol, "ener" )
        os.system( self.exe )
        self.parse_log( mol, "ener" )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        self.parse_log( mol, "grad" )
