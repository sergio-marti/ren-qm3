import  numpy
import  typing
import  os
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, fdsc: typing.IO,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0
        self.exe = "bash r.nwchem"
        self.inp = fdsc.read()
        self.__dq = numpy.zeros( mol.natm )
        # redistribute MM-charge on the remaining atoms of the group
        for i,j in self.lnk:
            if( j in self.grp ):
                self.__dq[self.grp[j]] += mol.chrg[j] / len( self.grp[j] )
        # ----------------------------------------------------------


    def mk_input( self, mol, run ):
        if( self.img ):
            cen = numpy.mean( mol.coor[self.sel], axis = 0 )
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
            s_mm = "set bq:max_nbq %d\n"%( len( self.nbn ) + 1 )
            s_mm += "bq units angstroms\n  force nwchem.mmgrad\n  load nwchem.mmchrg units angstroms format 1 2 3 4\nend"
            g = open( "nwchem.mmchrg", "wt" )
            for i in self.nbn:
                tmp = mol.coor[i].copy()
                if( self.img ):
                    tmp -= mol.boxl * numpy.round( ( tmp - cen ) / mol.boxl, 0 )
                g.write( "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] + self.__dq[i] ) )
            g.close()
        if( self.inp.lower().find( "dft" ) > -1 ):
            s_rn = "task dft"
        else:
            s_rn = "task scf"
        if( run == "grad" ):
            s_rn += " gradient"
        s_wf = ""
        if( os.access( "nwchem.movecs", os.R_OK ) ):
            s_wf = "vectors input nwchem.movecs"
        f = open( "nwchem.nw", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_guess", s_wf )
        buf = buf.replace( "qm3_charges", s_mm )
        f.write( buf )
        f.close()


    def parse_log( self, mol, run ):
        f = open( "nwchem.log", "rt" )
        l = f.readline()
        while( l != "" ):
            if( l.find( "Total " ) > -1 and l.find( " energy = " ) > -1 ):
                out = float( l.split()[-1] ) * self.ce
            elif( run == "grad" and l.find( "ENERGY GRADIENTS" ) > -1 ):
                f.readline(); f.readline(); f.readline()
                g = []
                for i in range( len( self.sel ) + len( self.lnk ) ):
                    g += [ float( j ) * self.cg for j in f.readline().strip().split()[-3:] ]
                qm3.engines.Link_grad( self.vla, g )
                k = 0
                for i in self.sel:
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += g[k]
                        k += 1
            l = f.readline()
        if( len( self.nbn ) > 0 and os.access( "nwchem.mmgrad", os.R_OK ) ):
            mol.grad[self.nbn] += numpy.loadtxt( "nwchem.mmgrad" ) * self.cg
        mol.func += out
        return( out )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        return( self.parse_log( mol, "grad" ) )

