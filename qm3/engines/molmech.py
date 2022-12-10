import  numpy
import  typing
import  inspect
import  qm3
import  qm3.data
import  qm3.utils._conn
import  qm3.engines._molmech
import  qm3.engines.mmres
import  os
import  pickle



class run( object ):
    def __init__( self, mol: object, ncpu: typing.Optional[int] = os.sysconf( 'SC_NPROCESSORS_ONLN' ) ):
        self.path     = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.ncpu     = ncpu
        self.cut_on   = -1
        self.cut_off  = -1
        self.cut_list = -1
        self.natm     = mol.natm
        self.nbnd     = []
        self.qmat     = [ False for i in range( mol.natm ) ]


    def initialize( self, mol: object,
            bond: typing.Optional[list] = [],
            angl: typing.Optional[list] = [],
            dihe: typing.Optional[list] = [],
            impr: typing.Optional[list] = [],
            qtyp: typing.Optional[bool] = True,
            qchg: typing.Optional[bool] = True ):
        """
        impr = [ [ central_i, j, k, l, kmb (kal/mol.rad^2), ref (deg) ], ... ]
        """
        if( len( bond ) > 0 ):
            self.bond = bond[:]
        else:
            self.bond = qm3.utils._conn.connectivity( self.ncpu, mol.anum, mol.coor )
        self.conn = [ [] for i in range( self.natm ) ]
        for i,j in self.bond:
            self.conn[i].append( j )
            self.conn[j].append( i )
        if( len( angl ) > 0 ):
            self.angl = angl[:]
        else:
            self.angl = qm3.engines._molmech.guess_angles( self )
        if( len( dihe ) > 0 ):
            self.dihe = dihe[:]
        else:
            self.dihe = qm3.engines._molmech.guess_dihedrals( self )
        self.impr = impr[:]
        if( qtyp ):
            self.guess_types( mol )
        if( qchg ):
            self.guess_charges( mol )
            
    
    def guess_types( self, mol: object ):
        """
        SYBYL atom types (kinda)
        http://www.sdsc.edu/CCMS/Packages/cambridge/pluto/atom_types.html

        uses FORMAL CHARGES (integers, fractional only for carboxylates) present in mol.chrg
        """
        def __any( lst_a, lst_b ):
            return( len( set( lst_a ).intersection( set( lst_b ) ) ) > 0 )
        mol.type = []
        # default to atomic symbol...
        for i in range( mol.natm ):
            mol.type.append( qm3.data.symbol[mol.anum[i]] )
            nb = len( self.conn[i] )
            if( mol.anum[i] == 1 ):
                if( mol.anum[self.conn[i][0]] == 6 ):
                    mol.type[i] = "Hn"
            elif( mol.anum[i] == 14 ):
                mol.type[i] = "Si.3"
            elif( mol.anum[i] == 15 ):
                mol.type[i] = "P.3"
            elif( mol.anum[i] in [ 6, 7, 8, 16 ] ):
                mol.type[i] += "_%d"%( len( self.conn[i] ) )
        # 2nd pass
        for i in range( mol.natm ):
            if( mol.type[i] in [ "C_2", "C_1" ] ):
                mol.type[i] = "C.1"
            elif( mol.type[i] == "C_4" ):
                mol.type[i] = "C.3"

            elif( mol.type[i] == "C_3" ):
                if( __any( [ "C_3", "C.ar" ], [ mol.type[j] for j in self.conn[i] ] ) ):
                    mol.type[i] = "C.ar"
                else:
                    if( __any( [ "O_1", "O.co2", "O.2" ], [ mol.type[j] for j in self.conn[i] ] ) ):
                        mol.type[i] = "C.co"
                    else:
                        mol.type[i] = "C.2"
            elif( mol.type[i] == "O_1" ):
                if( mol.type[self.conn[i][0]] in [ "C_3", "C.co" ] and mol.chrg[i] == -0.5 ):
                    mol.type[i] = "O.co2"
                elif( mol.chrg[i] == -1.0 ):
                    mol.type[i] = "O.x"
                else:
                    mol.type[i] = "O.2"
            elif( mol.type[i] == "O_2" ):
                if( 1 in [ mol.anum[j] for j in self.conn[i] ] ):
                    mol.type[i] = "O.h"
                else:
                    mol.type[i] = "O.3"
            elif( mol.type[i] == "N_4" ):
                mol.type[i] = "N.4"
            elif( mol.type[i] == "N_3" ):
                if( mol.chrg[i] == 1.0 ):
                    mol.type[i] = "N.pl"
                else:
                    mol.type[i] = "N.3"
            elif( mol.type[i] == "N_2" ):
                mol.type[i] = "N.2"
            elif( mol.type[i] == "N_1" ):
                mol.type[i] = "N.1"
            elif( mol.type[i] == "S_2" ):
                if( 1 in [ mol.anum[j] for j in self.conn[i] ] ):
                    mol.type[i] = "S.h"
                else:
                    mol.type[i] = "S.3"
            elif( mol.type[i] == "S_1" ):
                if( mol.chrg[i] == -1.0 ):
                    mol.type[i] = "S.x"
                else:
                    mol.type[i] = "S.2"
            elif( mol.type[i] == "S_3" and __any( [ "O_1", "O.2" ], [ mol.type[j] for j in self.conn[i] ] ) ):
                mol.type[i] = "S.o"
            elif( mol.type[i] == "S_4" and __any( [ "O_1", "O.2" ], [ mol.type[j] for j in self.conn[i] ] ) ):
                mol.type[i] = "S.o2"


    def guess_charges( self, mol: object, parm: typing.Optional[str] = "" ):
        """
        Electronegativity Equalization Method (B3LYP_6-311G_NPA.par) [10.1186/s13321-015-0107-1]

        uses FORMAL CHARGES (integers, fractional only for carboxylates) present in mol.chrg
        """
        if( parm != "" ):
            f = open( parm, "rt" )
        else:
            f = open( self.path + "molmech.eem", "rt" )
        kap = float( f.readline().strip() )
        prm = {}
        for l in f:
            t = l.strip().split()
            prm[t[0]] = [ float( t[1] ), float( t[2] ) ]
        f.close()
        mat = []
        vec = []
        for i in range( mol.natm ):
            for j in range( mol.natm ):
                if( j == i ):
                    mat.append( prm[mol.type[i]][1] )
                else:
                    mat.append( kap / qm3.utils.distance( mol.coor[i], mol.coor[j] ) )
            mat.append( -1 )
            vec.append( - prm[mol.type[i]][0] )
        mat += [ 1 ] * mol.natm + [ 0 ]
        vec.append( sum( mol.chrg ) )
        mat = numpy.array( mat ).reshape( ( mol.natm + 1, mol.natm + 1 ) )
        vec = numpy.array( vec ).reshape( ( mol.natm + 1, ) )
        mol.chrg = numpy.linalg.solve( mat, vec )[0:mol.natm]


    """
    def psf_read( self, mol, fname ):
        self.impr = []
        impr = []
        fd = qm3.fio.open_r( fname )
        fd.readline()
        fd.readline()
        for i in range( int( fd.readline().strip().split()[0] ) + 1 ):
            fd.readline()
        if( mol.natm == int( fd.readline().split()[0] ) ):
            mol.type = []
            mol.chrg = []
            mol.mass = []
            for i in range( mol.natm ):
                t = fd.readline().strip().split()
                mol.type.append( t[5] )
                mol.chrg.append( float( t[6] ) )
                mol.mass.append( float( t[7] ) )
            fd.readline()
            self.bond = []
            n = int( fd.readline().strip().split()[0] )
            while( len( self.bond ) < n ):
                t = [ int( i ) - 1 for i in fd.readline().strip().split() ]
                for i in range( len( t ) // 2 ):
                    self.bond.append( [ t[2*i], t[2*i+1] ] )
            self.conn = [ [] for i in range( self.natm ) ]
            for i,j in self.bond:
                self.conn[i].append( j )
                self.conn[j].append( i )
            fd.readline()
            self.angl = []
            n = int( fd.readline().strip().split()[0] )
            while( len( self.angl ) < n ):
                t = [ int( i ) - 1 for i in fd.readline().strip().split() ]
                for i in range( len( t ) // 3 ):
                    self.angl.append( [ t[3*i], t[3*i+1], t[3*i+2] ] )
            fd.readline()
            self.dihe = []
            n = int( fd.readline().strip().split()[0] )
            while( len( self.dihe ) < n ):
                t = [ int( i ) - 1 for i in fd.readline().strip().split() ]
                for i in range( len( t ) // 4 ):
                    self.dihe.append( [ t[4*i], t[4*i+1], t[4*i+2], t[4*i+3] ] )
            fd.readline()
            n = int( fd.readline().strip().split()[0] )
            while( len( impr ) < n ):
                t = [ int( i ) - 1 for i in fd.readline().strip().split() ]
                for i in range( len( t ) // 4 ):
                    impr.append( [ t[4*i], t[4*i+1], t[4*i+2], t[4*i+3] ] )
        else:
            print( "- Invalid number of atoms in PSF!" )
        qm3.fio.close( fd, fname )
        return( impr )
    """


    def load_parameters( self, mol: object, parm: typing.Optional[str] = "" ):
        if( parm != "" ):
            f = open( parm, "rt" )
        else:
            f = open( self.path + "molmech.prm", "rt" )
        out = True
        self.bond_data = []
        self.bond_indx = []
        self.angl_data = []
        self.angl_indx = []
        self.dihe_data = []
        self.dihe_indx = []
        self.impr_data = []
        self.impr_indx = []
        tmp_typ = {}
        cnt_bnd = 0
        tmp_bnd = {}
        cnt_ang = 0
        tmp_ang = {}
        cnt_dih = 0
        tmp_dih = {}
        cnt_imp = 0
        tmp_imp = {}
        for l in f:
            t = l.strip().split()
            if( len( t ) > 0 and t[0][0] != "#" ):
                if( len( t ) == 3 ):
                    tmp_typ[t[0]] = [ numpy.sqrt( float( t[1] ) * qm3.data.K2J ), float( t[2] ) ]
                elif( len( t ) == 4 ):
                    self.bond_data.append( [ float( t[2] ) * qm3.data.K2J, float( t[3] ) ] )
                    tmp_bnd["%s:%s"%( t[0], t[1] )] = cnt_bnd
                    cnt_bnd += 1
                elif( len( t ) == 5 ):
                    self.angl_data.append( [ float( t[3] ) * qm3.data.K2J, float( t[4] ) / qm3.data.R2D ] )
                    tmp_ang["%s:%s:%s"%( t[0], t[1], t[2] )] = cnt_ang
                    cnt_ang += 1
                elif( len( t ) >= 7 ):
                    tmp = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                    for i in range( 4, len( t ), 3 ):
                        n = int( t[i+1] ) - 1
                        if( n >= 0 and n < 6 ):
                            tmp[2*n]   = float( t[i]   ) * qm3.data.K2J
                            tmp[2*n+1] = float( t[i+2] ) / qm3.data.R2D
                    self.dihe_data.append( tmp[:] )
                    tmp_dih["%s:%s:%s:%s"%( t[0], t[1], t[2], t[3] )] = cnt_dih
                    cnt_dih += 1
        f.close()
        mol.epsi = []
        mol.rmin = []
        for i in range( mol.natm ):
            if( mol.type[i] in tmp_typ ):
                mol.epsi.append( tmp_typ[mol.type[i]][0] )
                mol.rmin.append( tmp_typ[mol.type[i]][1] )
            else:
                mol.epsi.append( None )
                mol.rmin.append( None )
                print( "- missing atom type [%s]: %d"%( mol.type[i], i+1 ) )
                out = False
        for i,j in self.bond:
            td = "%s:%s"%( mol.type[i], mol.type[j] )
            ti = "%s:%s"%( mol.type[j], mol.type[i] )
            if( td in tmp_bnd ):
                self.bond_indx.append( tmp_bnd[td] )
            elif( ti in tmp_bnd ):
                self.bond_indx.append( tmp_bnd[ti] )
            else:
                self.bond_indx.append( None )
                print( "- missing parameter [bond]: ", td )
                out = False
        for i,j,k in self.angl:
            td = "%s:%s:%s"%( mol.type[i], mol.type[j], mol.type[k] )
            ti = "%s:%s:%s"%( mol.type[k], mol.type[j], mol.type[i] )
            ts = "*:%s:*"%( mol.type[j] )
            if( td in tmp_ang ):
                self.angl_indx.append( tmp_ang[td] )
            elif( ti in tmp_ang ):
                self.angl_indx.append( tmp_ang[ti] )
            elif( ts in tmp_ang ):
                self.angl_indx.append( tmp_ang[ts] )
            else:
                self.angl_indx.append( None )
                print( "- missing parameter [angl]: ", td )
                out = False
        for i,j,k,l in self.dihe:
            td = "%s:%s:%s:%s"%( mol.type[i], mol.type[j], mol.type[k], mol.type[l] )
            ti = "%s:%s:%s:%s"%( mol.type[l], mol.type[k], mol.type[j], mol.type[i] )
            ts = "*:%s:%s:*"%( mol.type[j], mol.type[k] )
            tz = "*:%s:%s:*"%( mol.type[k], mol.type[j] )
            if( td in tmp_dih ):
                self.dihe_indx.append( tmp_dih[td] )
            elif( ti in tmp_dih ):
                self.dihe_indx.append( tmp_dih[ti] )
            elif( ts in tmp_dih ):
                self.dihe_indx.append( tmp_dih[ts] )
            elif( tz in tmp_dih ):
                self.dihe_indx.append( tmp_dih[tz] )
            else:
                self.dihe_indx.append( None )
                print( "- missing parameter [dihe]: ", td )
                out = False
        return( out )


    def define_QM( self, sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ) ):
        for i in numpy.argwhere( sele ).ravel():
            self.qmat[i] = True
        # delete QM-QM bonds
        for i in range( len( self.bond ) -1, -1, -1 ):
            if( self.qmat[self.bond[i][0]] and self.qmat[self.bond[i][1]] ):
                del self.bond[i]
                del self.bond_indx[i]
        # delete QM-QM-MM angles
        for i in range( len( self.angl ) -1, -1, -1 ):
            if( sum( [ self.qmat[j] for j in self.angl[i] ] ) >= 2 ):
                del self.angl[i]
                del self.angl_indx[i]
        # delete QM-QM-QM-MM dihedrals
        for i in range( len( self.dihe ) -1, -1, -1 ):
            if( sum( [ self.qmat[j] for j in self.dihe[i] ] ) >= 3 ):
                del self.dihe[i]
                del self.dihe_indx[i]
        # delete QM-QM-QM-MM impropers
        for i in range( len( self.impr ) -1, -1, -1 ):
            if( sum( [ self.qmat[j] for j in self.impr[i][0:4] ] ) >= 3 ):
                del self.impr[i]
        # delete QM-QM non_bonded
        for i in range( len( self.nbnd ) -1, -1, -1 ):
            if( self.qmat[self.nbnd[i][0]] and self.qmat[self.nbnd[i][1]] ):
                del self.nbnd[i]
        

    def update_nonbonded( self, mol ):
        self.nbnd = qm3.engines._molmech.update_non_bonded( self, mol )


    def __calculate( self, mol: object, epsilon: float, gradient: bool, qprint: bool ):
        self.actv = mol.actv.ravel().tolist()
        e_bond = 0.0
        if( len( self.bond ) > 0 ):
            e_bond = qm3.engines._molmech.ebond( self, mol, gradient )
        e_angl = 0.0
        if( len( self.angl ) > 0 ):
            e_angl = qm3.engines._molmech.eangle( self, mol, gradient )
        e_dihe = 0.0
        if( len( self.dihe ) > 0 ):
            e_dihe = qm3.engines._molmech.edihedral( self, mol, gradient )
        e_impr = 0.0
        if( len( self.impr ) > 0 ):
            bak = mol.func
            mol.func = 0.0
            for i in range( len( self.impr ) ):
                if( self.actv[self.impr[i][0]] or self.actv[self.impr[i][1]] or
                    self.actv[self.impr[i][2]] or self.actv[self.impr[i][3]] ):
                    qm3.engines.mmres.f_improper( mol, self.impr[i][4] * qm3.data.K2J, self.impr[i][5],
                        self.impr[i][0], self.impr[i][1], self.impr[i][2], self.impr[i][3], gradient )
            e_impr = mol.func
            mol.func = bak
        if( len( self.nbnd ) == 0 ):
            self.update_nonbonded( mol )
        e_elec, e_vdwl = qm3.engines._molmech.enonbonded( self, mol, gradient, epsilon )
        mol.func += e_bond + e_angl + e_dihe + e_impr + e_elec + e_vdwl
        if( qprint ):
            print( "ETot:", e_bond + e_angl + e_dihe + e_impr + e_elec + e_vdwl, "_kJ/mol" )
            print( "   Bond:%18.4lf   Angl:%18.4lf   Dihe:%18.4lf"%( e_bond, e_angl, e_dihe ) )
            print( "   Impr:%18.4lf   Elec:%18.4lf   VdWl:%18.4lf"%( e_impr, e_elec, e_vdwl ) )


    def get_func( self, mol: object,
            epsilon: typing.Optional[float] = 1.0,
            qprint: typing.Optional[bool] = False ):
        self.__calculate( mol, epsilon, False, qprint )

    def get_grad( self, mol: object,
            epsilon: typing.Optional[float] = 1.0,
            qprint: typing.Optional[bool] = False ):
        self.__calculate( mol, epsilon, True, qprint )


    def system_write( self, fdsc: typing.IO ):
        pickle.dump( self.natm, fdsc )
        pickle.dump( self.bond, fdsc )
        pickle.dump( self.conn, fdsc )
        pickle.dump( self.angl, fdsc )
        pickle.dump( self.dihe, fdsc )
        pickle.dump( self.impr, fdsc )
        pickle.dump( self.bond_data, fdsc )
        pickle.dump( self.bond_indx, fdsc )
        pickle.dump( self.angl_data, fdsc )
        pickle.dump( self.angl_indx, fdsc )
        pickle.dump( self.dihe_data, fdsc )
        pickle.dump( self.dihe_indx, fdsc )
        pickle.dump( self.impr_data, fdsc )
        pickle.dump( self.impr_indx, fdsc )

    def system_read( self, fdsc: typing.IO ):
        self.natm = pickle.load( fdsc )
        self.bond = pickle.load( fdsc )
        self.conn = pickle.load( fdsc )
        self.angl = pickle.load( fdsc )
        self.dihe = pickle.load( fdsc )
        self.impr = pickle.load( fdsc )
        self.bond_data = pickle.load( fdsc )
        self.bond_indx = pickle.load( fdsc )
        self.angl_data = pickle.load( fdsc )
        self.angl_indx = pickle.load( fdsc )
        self.dihe_data = pickle.load( fdsc )
        self.dihe_indx = pickle.load( fdsc )
        self.impr_data = pickle.load( fdsc )
        self.impr_indx = pickle.load( fdsc )
