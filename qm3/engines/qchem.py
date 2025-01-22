import  numpy
import  typing
import  os
import  re
import  qm3.data
import  qm3.utils
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, fdsc: typing.IO,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0
        self.exe = "bash r.qchem"
        self.inp = fdsc.read()
        self.pat = re.compile( "The QM part of the energy is[\\ ]+([0-9\\.\\-]+)" )
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
        s_mm = "$external_charges\n"
        for i in self.nbn:
            tmp = mol.coor[i].copy()
            if( self.img ):
                tmp -= mol.boxl * numpy.round( ( tmp - cen ) / mol.boxl, 0 )
            s_mm += "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] + self.__dq[i] )
        s_mm += "$end"
        s_rn = "single_point"
        if( run == "grad" ):
            s_rn = "force"
        s_wf = ""
        if( os.path.isdir( "qchem.tmp" ) ):
            s_wf = "scf_guess read"
        f = open( "qchem.inp", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_guess", s_wf )
        if( len( self.nbn ) > 0 ):
            buf = buf.replace( "qm3_charges", s_mm )
        f.write( buf )
        f.close()


    def parse_log( self, mol, run ):
        f = open( "qchem.log", "rt" )
        out = float( self.pat.findall( f.read() )[0] ) * self.ce
        f.close()
        mol.func += out
        if( run == "grad" ):
            f = open( "efield.dat", "rt" )
            if( len( self.nbn ) > 0 ):
                for i in self.nbn:
                    t = f.readline().strip().split()
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += - self.cg * mol.chrg[i] * float( t[j] )
            g = []
            for i in range( len( self.sel ) + len( self.lnk ) ):
                g += [ float( j ) * self.cg for j in f.readline().strip().split() ]
            f.close()
            qm3.engines.Link_grad( self.vla, g )
            k = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[k]
                    k += 1
        return( out )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        return( self.parse_log( mol, "grad" ) )

