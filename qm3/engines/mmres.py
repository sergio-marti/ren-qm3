import  math
import  numpy
import  typing
import  qm3.data
#import  collections


def f_distance( mol: object, kumb: float, xref: float, a_i: int, a_j: int,
        skip_LE: typing.Optional[float] = 0.0,
        skip_BE: typing.Optional[float] = 9.e99,
        grad: typing.Optional[bool] = False ) -> float:
    """
    e_bond = force_constant / 2 * ( distance - reference )^2

    force_constant [kJ/mol.A^2]
    reference [A]
    """
    dr = mol.coor[a_i] - mol.coor[a_j]
    vv = numpy.linalg.norm( dr )
    df = kumb * ( vv - xref )
    oo = 0.0
    if( vv >= skip_LE and vv <= skip_BE ):
        oo = 0.5 * df * ( vv - xref )
        mol.func += oo
        if( grad ):
            df /= vv
            mol.grad[a_i] += dr * df
            mol.grad[a_j] -= dr * df
    return( ( oo, vv ) )


def f_angle( mol: object, kumb: float, xref: float, a_i: int, a_j: int, a_k: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    e_angle = force_constant / 2 * ( angle - reference )^2

    force_constant [kJ/mol.rad^2]
    reference [rad]
    return_value [deg]
    """
    dij = mol.coor[a_i] - mol.coor[a_j]
    rij = numpy.linalg.norm( dij )
    dij /= rij
    dkj = mol.coor[a_k] - mol.coor[a_j]
    rkj = numpy.linalg.norm( dkj )
    dkj /= rkj
    dot = numpy.dot( dij, dkj )
    dot = min( 1.0, max( -1.0, dot ) )
    vv  = math.acos( dot )
    dv  = ( vv - xref )
    df  = kumb * dv
    oo  = 0.5 * df * dv
    mol.func += oo
    if( grad ):
        dx  = - 1.0 / math.sqrt( 1.0 - dot * dot )
        df *= dx
        dti = ( dkj - dot * dij ) / rij
        dtk = ( dij - dot * dkj ) / rkj
        dtj = - ( dti + dtk )
        mol.grad[a_i] += df * dti
        mol.grad[a_j] += df * dtj
        mol.grad[a_k] += df * dtk
    return( ( oo, vv * qm3.data.R2D ) )


def f_dihedral( mol: object, data: list, a_i: int, a_j: int, a_k: int, a_l: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    e_dihedral = force_constant * ( 1 + cos( periodicity * angle - displacement ) )

    force_constant [kJ/mol]
    displacement [rad]

    data = [ frc_per=1, dsp_per=1, frc_per=2, dsp_per=2, ..., frc_per=6, dsp_per=6 ]

    can be used for umbrella sampling (in radians) with:
        frc_per[1] = kumb
        dsp_per[1] = pi + xref
    """
    dji = mol.coor[a_j] - mol.coor[a_i]
    dkj = mol.coor[a_k] - mol.coor[a_j]
    dlk = mol.coor[a_l] - mol.coor[a_k]
    vt  = numpy.cross( dji, dkj )
    vu  = numpy.cross( dkj, dlk )
    vtu = numpy.cross( vt, vu )
    rt2 = numpy.dot( vt, vt )
    ru2 = numpy.dot( vu, vu )
    rtu = math.sqrt( rt2 * ru2 )
    rkj = numpy.linalg.norm( dkj )
    cs1 = numpy.dot( vt, vu ) / rtu
    cs1 = min( 1.0, max( -1.0, cs1 ) )
    sn1 = numpy.dot( dkj, vtu ) / ( rkj * rtu )
    cs2 = cs1 * cs1 - sn1 * sn1
    sn2 = 2.0 * cs1 * sn1
    cs3 = cs1 * cs2 - sn1 * sn2
    sn3 = cs1 * sn2 + sn1 * cs2
    cs4 = cs1 * cs3 - sn1 * sn3
    sn4 = cs1 * sn3 + sn1 * cs3
    cs5 = cs1 * cs4 - sn1 * sn4
    sn5 = cs1 * sn4 + sn1 * cs4
    cs6 = cs1 * cs5 - sn1 * sn5
    sn6 = cs1 * sn5 + sn1 * cs5
    dph = 0.0
    out = 0.0
    if( data[0] != 0.0 ):
        cd  = numpy.cos( data[1] )
        sd  = numpy.sin( data[1] )
        dph += data[0] * ( cs1 * sd - sn1 * cd )
        out += data[0] * ( 1.0 + cs1 * cd + sn1 * sd )
    if( data[2] != 0.0 ):
        cd  = numpy.cos( data[3] )
        sd  = numpy.sin( data[3] )
        dph += data[2] * 2.0 * ( cs2 * sd - sn2 * cd )
        out += data[2] * ( 1.0 + cs2 * cd + sn2 * sd )
    if( data[4] != 0.0 ):
        cd  = numpy.cos( data[5] )
        sd  = numpy.sin( data[5] )
        dph += data[4] * 3.0 * ( cs3 * sd - sn3 * cd )
        out += data[4] * ( 1.0 + cs3 * cd + sn3 * sd )
    if( data[6] != 0.0 ):
        cd  = numpy.cos( data[7] )
        sd  = numpy.sin( data[7] )
        dph += data[6] * 4.0 * ( cs4 * sd - sn4 * cd )
        out += data[6] * ( 1.0 + cs4 * cd + sn4 * sd )
    if( data[8] != 0.0 ):
        cd  = numpy.cos( data[9] )
        sd  = numpy.sin( data[9] )
        dph += data[8] * 5.0 * ( cs5 * sd - sn5 * cd )
        out += data[8] * ( 1.0 + cs5 * cd + sn5 * sd )
    if( data[10] != 0.0 ):
        cd  = numpy.cos( data[11] )
        sd  = numpy.sin( data[11] )
        dph += data[10] * 6.0 * ( cs6 * sd - sn6 * cd )
        out += data[10] * ( 1.0 + cs6 * cd + sn6 * sd )
    mol.func += out
    if( grad ):
        dki = mol.coor[a_k] - mol.coor[a_i]
        dlj = mol.coor[a_l] - mol.coor[a_j]
        dvt = numpy.cross( vt, dkj ) / ( rt2 * rkj )
        dvu = numpy.cross( vu, dkj ) / ( ru2 * rkj )
        mol.grad[a_i] += dph * numpy.cross( dvt, dkj )
        mol.grad[a_j] += dph * ( numpy.cross( dki, dvt ) - numpy.cross( dvu, dlk ) )
        mol.grad[a_k] += dph * ( numpy.cross( dvt, dji ) - numpy.cross( dlj, dvu ) )
        mol.grad[a_l] -= dph * numpy.cross( dvu, dkj )
    ang = qm3.data.R2D * math.acos( cs1 )
    if( sn1 <= 0.0 ):
        ang = -ang
    return( ( out, ang ) )


def f_improper( mol: object, kumb: float, xref: float, a_i: int, a_j: int, a_k: int, a_l: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    e_improper = force_constant / 2 * ( angle - reference )^2

    force_constant [kJ/mol.rad^2]
    reference [deg]
    a_i should be central atom
    """
    dji = mol.coor[a_j] - mol.coor[a_i]
    dkj = mol.coor[a_k] - mol.coor[a_j]
    dlk = mol.coor[a_l] - mol.coor[a_k]
    vt  = numpy.cross( dji, dkj )
    vu  = numpy.cross( dkj, dlk )
    vtu = numpy.cross( vt, vu )
    rt2 = numpy.sum( vt * vt )
    ru2 = numpy.sum( vu * vu )
    rtu = math.sqrt( rt2 * ru2 )
    rkj = numpy.linalg.norm( dkj )
    cos = numpy.sum( vt * vu ) / rtu
    cos = min( 1.0, max( -1.0, cos ) )
    sin = numpy.sum( dkj * vtu ) / ( rkj * rtu )
    ang = qm3.data.R2D * math.acos( cos )
    if( sin <= 0.0 ):
        ang = -ang
    if( math.fabs( ang + xref ) < math.fabs( ang - xref ) ):
        xref = -xref
    dt  = ang - xref
    while( dt >  180.0 ):
        dt -= 360.0
    while( dt < -180.0 ):
        dt += 360.0
    dt /= qm3.data.R2D
    out = 0.5 * kumb * dt * dt
    mol.func += out
    if( grad ):
        dph = kumb * dt
        dki = mol.coor[a_k] - mol.coor[a_i]
        dlj = mol.coor[a_l] - mol.coor[a_j]
        dvt = numpy.cross( vt, dkj ) / ( rt2 * rkj )
        dvu = numpy.cross( vu, dkj ) / ( ru2 * rkj )
        mol.grad[a_i] += dph * numpy.cross( dvt, dkj )
        mol.grad[a_j] += dph * ( numpy.cross( dki, dvt ) - numpy.cross( dvu, dlk ) )
        mol.grad[a_k] += dph * ( numpy.cross( dvt, dji ) - numpy.cross( dlj, dvu ) )
        mol.grad[a_l] -= dph * numpy.cross( dvu, dkj )
    return( ( out, ang ) )


class distance( object ):
    def __init__( self, kumb: float, xref: float, indx: list,
            skip_LE: typing.Optional[float] = 0.0,
            skip_BE: typing.Optional[float] = 9.e99 ):
        self.kumb = kumb
        self.xref = xref
        self.indx = indx[:]
        self.skpL = skip_LE
        self.skpB = skip_BE

    def get_func( self, mol: object ):
        return( f_distance( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.skpL, self.skpB ) )

    def get_grad( self, mol: object ):
        return( f_distance( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.skpL, self.skpB, True ) )


class angle( object ):
    def __init__( self, kumb: float, xref: float, indx: list ):
        self.kumb = kumb
        self.xref = xref / qm3.data.R2D
        self.indx = indx[:]

    def get_func( self, mol: object ):
        return( f_angle( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2] ) )

    def get_grad( self, mol: object ):
        return( f_angle( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], True ) )


class dihedral( object ):
    def __init__( self, data: dict, indx: list ):
        """
    data = {  periodicity: [ force_constant [kJ/mol], displacement [degrees] ], ... }

    X - C_sp3 - C_sp3 - X   =>  { 3: [ 0.8159, 0.0 ] }

    valid periodicities = [ 1 : 6 ]
        """
        self.data = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
        for i in range( 6 ):
            if( i+1 in data ):
                self.data[2*i]   = data[i+1][0]
                self.data[2*i+1] = data[i+1][1] / qm3.data.R2D
        self.indx = indx[:]

    def get_func( self, mol: object ):
        return( f_dihedral( mol, self.data, self.indx[0], self.indx[1], self.indx[2], self.indx[3] ) )

    def get_grad( self, mol: object ):
        return( f_dihedral( mol, self.data, self.indx[0], self.indx[1], self.indx[2], self.indx[3], True ) )


class improper( object ):
    def __init__( self, kumb: float, xref: float, indx: list ):
        self.kumb = kumb
        self.xref = xref
        self.indx = indx[:]

    def get_func( self, mol: object ):
        return( f_improper( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], self.indx[3] ) )

    def get_grad( self, mol: object ):
        return( f_improper( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], self.indx[3], True ) )


class multiple_distance( object ):
    def __init__( self, kumb: float, xref: float, indx: list, weigth: numpy.array ):
        """
    multiple_distance = force_constant / 2 * ( value - reference )^2

    value = SUM weigth_i * distance_i

    force_constant [kJ/mol.A^2]
    reference [A]
        """
        if( len( weigth ) * 2 != len( indx ) ):
            print( "- restraints.multiple_distance: Number of ATOMS should be TWICE the number of WEIGHTS!" )
            return( None )
        self.kumb = kumb
        self.xref = xref
        self.indx = indx[:]
        self.weig = weigth.copy()
        self.size = len( weigth )

    def get_func( self, mol: object ):
        rr = numpy.zeros( self.size, dtype=numpy.float64 )
        for i in range( self.size ):
            rr[i] = numpy.linalg.norm( mol.coor[self.indx[2*i]] - mol.coor[self.indx[2*i+1]] )
        vv = numpy.sum( rr * self.weig )
        oo = 0.5 * self.kumb * ( vv - self.xref ) * ( vv - self.xref )
        mol.func += oo
        return( ( oo, vv ) )

    def get_grad( self, mol: object ):
        dr = numpy.zeros( ( self.size, 3 ), dtype=numpy.float64 )
        rr = numpy.zeros( self.size, dtype=numpy.float64 )
        for i in range( self.size ):
            dr[i] = mol.coor[self.indx[2*i]] - mol.coor[self.indx[2*i+1]]
            rr[i] = numpy.linalg.norm( dr[i] )
        vv = numpy.sum( rr * self.weig )
        df = self.kumb * ( vv - self.xref )
        oo = 0.5 * df * ( vv - self.xref )
        mol.func += oo
        for i in range( self.size ):
            tt = self.weig[i] * df / rr[i]
            mol.grad[self.indx[2*i]] += tt * dr[i]
            mol.grad[self.indx[2*i+1]] -= tt * dr[i]
        return( ( oo, vv ) )


class tether( object ):
    """
    thether = force_constant / 2 * SUM ( cartesian - reference )^2

    force_constant [kJ/mol.A^2]
    reference [A]
    """
    def __init__( self, mol: object, kumb: float,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ) ):
        self.kumb = kumb
        if( sele.sum() > 0 ):
            self.sele = sele.reshape( ( mol.natm, 1 ) ) * 1.0
        else:
            self.sele = numpy.ones( ( mol.natm, 1 ) )
        self.cref = mol.coor.copy()

    def get_func( self, mol: object ):
        dr = ( mol.coor - self.cref ) * self.sele
        oo = 0.5 * self.kumb * numpy.sum( dr * dr )
        mol.func += oo
        return( oo )

    def get_grad( self, mol: object ):
        dr = ( mol.coor - self.cref ) * self.sele
        oo = 0.5 * self.kumb * numpy.sum( dr * dr )
        mol.func += oo
        mol.grad += self.kumb * dr
        return( oo )


class colvar_s( object ):
    """
    kumb: kJ / ( mol Angs^2 amu )
    xref: Ang amu^0.5
    -------------------------------------
    ncrd        nwin
    atom_1,i    atom_1,j
    ...         ...
    atom_n,i    atom_n,j
    -------------------------------------
    ref_1,1     ref_1,nc
    ...         ...
    ref_nw,1    ref_nw,nc
    -------------------------------------
    met_1,(1,1)   met_1,(nc,nc) [by rows]
    ...           ...
    met_nw,(1,1)  met_nw,(nc,nc)
    -------------------------------------

    x( %zeta  ) approx {sum from{i=0} to{N-1} {i %delta_z e^{-{{lline %zeta - z rline}over %delta_z}}}} over
    {sum from{i=0} to{N-1} { e^{-{{lline %zeta - z rline}over %delta_z}}}}
    ~~~~
    %delta_z = langle lline x_{i+1} - x_{i} rline rangle = L over{N - 1}
    newline
    lline %zeta - z rline = left[ (%zeta - z)^T M^{-1} (%zeta - z) right]^{1 over 2}
    ~~~~
    M_{i,j}=sum from{k=1} to{3n} {{partial %zeta_i}over{partial x_k} 1 over m_k {partial %zeta_j}over{partial x_k}}

    J. Comput. Chem. v35, p1672 (2014) [10.1002/jcc.23673]
    J. Phys. Chem. A v121, p9764 (2017) [10.1021/acs.jpca.7b10842]
    WIREs Comput. Mol. Sci. v8 (2018) [10.1002/wcms.1329]
    """
    def __init__( self, mol: object, kumb: float, xref: float,
            str_cnf: typing.IO,
            str_crd: typing.IO,
            str_met: typing.IO,
            masses: typing.Optional[bool] = False ):
        self.xref = xref
        self.kumb = kumb
        self.qmas = masses
        # parse config
        tmp = str_cnf.readline().strip().split()
        self.ncrd = int( tmp[0] )
        self.ncr2 = self.ncrd * self.ncrd
        self.nwin = int( tmp[1] )
        self.atom = []
        self.jidx = {}
        for i in range( self.ncrd ):
            tmp = [ int( j ) for j in str_cnf.readline().strip().split() ]
            self.atom.append( ( tmp[0], tmp[1] ) )
            self.jidx[tmp[0]] = True
            self.jidx[tmp[1]] = True
#        self.jidx = collections.OrderedDict( { jj: ii for ii,jj in enumerate( sorted( self.jidx ) ) } )
#        self.xdij = collections.OrderedDict( { self.jidx[ii]: ii for ii in self.jidx } )
        self.jidx = { jj: ii for ii,jj in enumerate( sorted( self.jidx ) ) }
        self.xdij = { self.jidx[ii]: ii for ii in self.jidx }
        self.jcol = 3 * len( self.jidx )
        # load previous equi-distributed string
        self.rcrd = numpy.array( [ float( i ) for i in str_crd.read().strip().split() ], dtype=numpy.float64 )
        self.rcrd.shape = ( self.nwin, self.ncrd )
        # load previous string metrics (default to identity...)
        try:
            self.rmet = numpy.array( [ float( i ) for i in str_met.read().strip().split() ], dtype=numpy.float64 )
            self.rmet.shape = ( self.nwin, self.ncr2 )
        except:
            self.rmet = numpy.zeros( ( self.nwin, self.ncr2 ), dtype=numpy.float64 )
            self.rmet[0:self.nwin,:] = numpy.eye( self.ncrd ).ravel()
        # get the arc of the current string
        self.arcl = numpy.zeros( self.nwin, dtype=numpy.float64 )
        for i in range( 1, self.nwin ):
            vec = self.rcrd[i] - self.rcrd[i-1]
            vec.shape = ( self.ncrd, 1 )
            mat = 0.5 * ( self.rmet[i] + self.rmet[i-1] )
            mat.shape = ( self.ncrd, self.ncrd )
            mat = numpy.linalg.inv( mat )
            self.arcl[i] = math.sqrt( numpy.dot( vec.T, numpy.dot( mat, vec ) ) )
        self.delz = self.arcl.sum() / float( self.nwin - 1.0 )
        if( self.qmas ):
            print( "Colective variable s range: [%.3lf - %.3lf: %.6lf] _Ang amu^0.5"%( 0.0, self.arcl.sum(), self.delz ) )
        else:
            print( "Colective variable s range: [%.3lf - %.3lf: %.6lf] _Ang"%( 0.0, self.arcl.sum(), self.delz ) )
        # store inverse (constant) metrics
        for i in range( self.nwin ):
            self.rmet[i] = numpy.linalg.inv( self.rmet[i].reshape( ( self.ncrd, self.ncrd ) ) ).ravel()
        # store the square root of the masses
        if( self.qmas ):
            self.sqms = numpy.sqrt( mol.mass[list( self.jidx.keys() )] )
        else:
            self.sqms = numpy.ones( len( self.jidx ), dtype=numpy.float64 )
        self.sqms = numpy.column_stack( ( self.sqms, self.sqms, self.sqms ) ).reshape( ( self.jcol // 3, 3 ) )


    def get_jaco( self, mol: object ) -> tuple:
        ccrd = numpy.zeros( self.ncrd, dtype=numpy.float64 )
        jaco = numpy.zeros( ( self.ncrd, self.jcol ), dtype=numpy.float64 )
        for i in range( self.ncrd ):
            ai = self.atom[i][0]
            aj = self.atom[i][1]
            rr = mol.coor[aj] - mol.coor[ai]
            ccrd[i] = numpy.linalg.norm( rr )
            for j in [0, 1, 2]:
                jaco[i,3*self.jidx[ai]+j] -= rr[j] / ccrd[i]
                jaco[i,3*self.jidx[aj]+j] += rr[j] / ccrd[i]
        return( ccrd, jaco )


    def metrics( self, mol: object ) -> numpy.array:
        ccrd, jaco = self.get_jaco( mol )
        if( self.qmas ):
            mass = mol.mass[list( self.jidx.keys() )]
        else:
            mass = numpy.ones( len( self.jidx ), dtype=numpy.float64 )
        mass = numpy.column_stack( ( mass, mass, mass ) ).reshape( self.jcol )
        cmet = numpy.zeros( ( self.ncrd, self.ncrd ), dtype=numpy.float64 )
        for i in range( self.ncrd ):
            for j in range( i, self.ncrd ):
                cmet[i,j] = numpy.sum( jaco[i,:] * jaco[j,:] / mass )
                cmet[j,i] = cmet[i,j]
        return( cmet )


    def get_func( self, mol: object ) -> numpy.array:
        ccrd, jaco = self.get_jaco( mol )
        cdst = numpy.zeros( self.nwin, dtype=numpy.float64 )
        for i in range( self.nwin ):
            vec = ccrd - self.rcrd[i]
            vec.shape = ( self.ncrd, 1 )
            mat = self.rmet[i]
            mat.shape = ( self.ncrd, self.ncrd )
            cdst[i] = math.sqrt( numpy.dot( vec.T, numpy.dot( mat, vec ) ) )
        cexp = numpy.exp( - cdst / self.delz )
        cval = self.delz * numpy.sum( numpy.arange( self.nwin, dtype=numpy.float64 ) * cexp ) / cexp.sum()
        out  = 0.5 * self.kumb * math.pow( cval - self.xref, 2.0 )
        mol.func += out
        return( ( out, cval, ccrd ) )


    def get_grad( self, mol: object ) -> numpy.array:
        ccrd, jaco = self.get_jaco( mol )
        cdst = numpy.zeros( self.nwin, dtype=numpy.float64 )
        jder = numpy.zeros( ( self.nwin, self.jcol ), dtype=numpy.float64 )
        for i in range( self.nwin ):
            vec = ccrd - self.rcrd[i]
            vec.shape = ( self.ncrd, 1 )
            mat = self.rmet[i]
            mat.shape = ( self.ncrd, self.ncrd )
            mat = numpy.dot( mat, vec )
            cdst[i] = math.sqrt( numpy.dot( vec.T, mat ) )
            tmp = numpy.dot( vec.T, numpy.dot( self.rmet[i].reshape( ( self.ncrd, self.ncrd ) ), jaco ) )
            jder[i] = 0.5 * ( numpy.dot( mat.T, jaco ) + tmp ).ravel() / cdst[i]
        cexp = numpy.exp( - cdst / self.delz )
        sumn = self.delz * numpy.sum( numpy.arange( self.nwin, dtype=numpy.float64 ) * cexp )
        sumd = cexp.sum()
        cval = sumn / sumd
        diff = self.kumb * ( cval - self.xref )
        out  = 0.5 * diff * ( cval - self.xref )
        mol.func += out
        sder = numpy.zeros( self.jcol, dtype=numpy.float64 )
        for i in range( self.jcol ):
#            sder[i] = diff * numpy.sum( jder[:,i] * cexp * ( cval / self.delz - numpy.arange( self.nwin, dtype=numpy.float64 ) ) ) / sumd
            for j in range( self.nwin ):
                sder[i] += diff * jder[j,i] * ( cval / self.delz - j ) * cexp[j] / sumd
#        tmp = self.jcol // 3
#        sder.shape = ( tmp, 3 )
#        for i in range( tmp ):
#            mol.grad[self.xdij[i]] += sder[i]
        sder.shape = ( self.jcol // 3, 3 )
        mol.grad[list( self.jidx.keys() ),:] += sder * self.sqms
        return( ( out, cval, ccrd ) )
