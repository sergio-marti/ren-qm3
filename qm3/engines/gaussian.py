import  numpy
import  typing
import  os
import  qm3.data
import  qm3.engines


class run( qm3.engines.template ):
    def __init__( self, mol: object, fdsc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            link: typing.Optional[list] = [] ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.exe = "bash r.gauss"
        self.inp = fdsc.read()
        self.gmm = ( self.inp.lower().find( "prop=(field,read)" ) > -1 )
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
        s_nq = ""
        for i in self.nbn:
            tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
            s_mm += "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] )
            s_nq += "%20.10lf%20.10lf%20.10lf\n"%( tmp[0], tmp[1], tmp[2] )
        s_rn = ""
        if( run == "grad" ):
            s_rn = "force"
#        elif( run == "hess" ):
#            s_rn = "freq=noraman cphf(maxinv=10000)"
        s_wf = ""
        if( os.access( "gauss.chk", os.R_OK ) ):
            s_wf = "guess=(read)"
        f = open( "gauss.com", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_guess", s_wf )
        buf = buf.replace( "qm3_charges", s_mm[:-1] )
        buf = buf.replace( "qm3_field", s_nq[:-1] )
        f.write( buf )
        f.write( "\n\n\n\n\n" )
        f.close()


    def parse_log( self, mol, run ):
        fd = open( "Test.FChk", "rt" )
        l = fd.readline()
        while( l != "" ):
            if( l[0:12] == "Total Energy" ):
                mol.func += float( l.split()[3] ) * self.ce
            if( run in [ "grad", "hess" ] and l[0:18] == "Cartesian Gradient" ):
                i = int( l.split()[-1] )
                j = int( i // 5 ) + ( i%5 != 0 )
                i = 0
                g = []
                while( i < j ):
                    l = fd.readline()
                    for itm in l.split():
                        g.append( float( itm ) * self.cg )
                    i += 1
                qm3.engines.Link_grad( self.vla, g )
                k = 0
                for i in self.sel:
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += g[k]
                        k += 1
# --------------------------------------------------------------------------------
#                # read hessian (columns)
#                if( run == "hess" ):
#                    l = fd.readline()
#                    i = int( l.split()[-1] )
#                    j = int( i // 5 ) + ( i % 5 != 0 )
#                    i = 0
#                    h = []
#                    while( i < j ):
#                        l = fd.readline()
#                        for itm in l.split():
#                            h.append( float( itm ) * self.ch )
#                        i += 1
#                    # truncate LAs and swap hessian (cols>>rows)
#                    i = 3 * len( self.sel )
#                    j = i * ( i + 1 ) // 2
#                    t = qm3.maths.matrix.from_upper_diagonal_columns( h[0:j], i )
#                    for j in range( i * i ):
#                        mol.hess[j] += t[j]
# --------------------------------------------------------------------------------
            if( l[0:11] == "ESP Charges" ):
                i = int( l.split()[-1] )
                j = int( i // 5 ) + ( i % 5 != 0 )
                i = 0
                k = 0
                while( i < j ):
                    l = fd.readline()
                    for itm in l.split():
                        if( k < len( self.sel ) ):
                            mol.chrg[self.sel[k]] = float( itm )
                            k += 1
                    i += 1
            l = fd.readline()
        fd.close()
        if( len( self.nbn ) > 0 ):
            fd = open( "gauss.log", "rt" )
            l = fd.readline()
            fl = True
            while( l != "" and fl ):
                if( l[0:29] == " Self energy of the charges =" ):
                    fl = False
                    mol.func -= float( l.split()[-2] ) * self.ce
                l = fd.readline()
            if( run in [ "grad", "hess" ] and self.gmm ):
                fl = True
                while( l != "" and fl ):
                    if( l.strip() == "Potential          X             Y             Z" ):
                        fl = False
                        for i in range( 1 + len( self.sel ) + len( self.lnk ) ):
                            fd.readline()
                        for i in self.nbn:
                            t = fd.readline().split()[2:]
                            for j in [0, 1, 2]:
                                mol.grad[i,j] += - self.cg * mol.chrg[i] * float( t[j] )
                    l = fd.readline()
            fd.close()
        os.unlink( "Test.FChk" )


    def get_func( self, mol ):
        self.mk_input( mol, "ener" )
        os.system( self.exe )
        self.parse_log( mol, "ener" )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        self.parse_log( mol, "grad" )
