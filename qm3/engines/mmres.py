import  numpy
import  typing
import  qm3.data


def f_distance( mol: object, kumb: float, xref: float, a_i: int, a_j: int,
        skip_LE: typing.Optional[float] = 0.0,
        skip_BE: typing.Optional[float] = 9.e99,
        grad: typing.Optional[bool] = False ) -> float:
    """
    bond = force_constant / 2 * ( distance - reference )^2

    force_constant [kJ/mol.A^2]
    reference [A]
    """
    dr = mol.coor[a_i] - mol.coor[a_j]
    vv = numpy.linalg.norm( dr )
    df = kumb * ( vv - xref )
    if( vv >= skip_LE and vv <= skip_BE ):
        mol.func += 0.5 * df * ( vv - xref )
        if( grad ):
            df /= vv
            mol.grad[a_i] += dr * df
            mol.grad[a_j] -= dr * df
    return( vv )


def f_angle( mol: object, kumb: float, xref: float, a_i: int, a_j: int, a_k: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    angle = force_constant / 2 * ( angle - reference )^2

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
    vv  = numpy.arccos( dot )
    dv  = ( vv - xref )
    df  = kumb * dv
    mol.func += 0.5 * df * dv
    if( grad ):
        dx  = - 1.0 / numpy.sqrt( 1.0 - dot * dot )
        df *= dx
        dti = ( dkj - dot * dij ) / rij
        dtk = ( dij - dot * dkj ) / rkj
        dtj = - ( dti + dtk )
        mol.grad[a_i] += df * dti
        mol.grad[a_j] += df * dtj
        mol.grad[a_k] += df * dtk
    return( vv * qm3.data.R2D )


def f_dihedral( mol: object, data: list, a_i: int, a_j: int, a_k: int, a_l: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    dihedral = force_constant * ( 1 + cos( periodicity * angle - displacement ) )

    force_constant [kJ/mol]
    displacement [rad]

    data = [ frc_per=1, dsp_per=1, frc_per=2, dsp_per=2, ..., frc_per=6, dsp_per=6 ]
    """
    dji = mol.coor[a_j] - mol.coor[a_i]
    dkj = mol.coor[a_k] - mol.coor[a_j]
    dlk = mol.coor[a_l] - mol.coor[a_k]
    vt  = numpy.cross( dji, dkj )
    vu  = numpy.cross( dkj, dlk )
    vtu = numpy.cross( vt, vu )
    rt2 = numpy.dot( vt, vt )
    ru2 = numpy.dot( vu, vu )
    rtu = numpy.sqrt( rt2 * ru2 )
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
    if( data[0] != 0.0 ):
        cd  = numpy.cos( data[1] )
        sd  = numpy.sin( data[1] )
        dph += data[0] * ( cs1 * sd - sn1 * cd )
        mol.func += data[0] * ( 1.0 + cs1 * cd + sn1 * sd )
    if( data[2] != 0.0 ):
        cd  = numpy.cos( data[3] )
        sd  = numpy.sin( data[3] )
        dph += data[2] * 2.0 * ( cs2 * sd - sn2 * cd )
        mol.func += data[2] * ( 1.0 + cs2 * cd + sn2 * sd )
    if( data[4] != 0.0 ):
        cd  = numpy.cos( data[5] )
        sd  = numpy.sin( data[5] )
        dph += data[4] * 3.0 * ( cs3 * sd - sn3 * cd )
        mol.func += data[4] * ( 1.0 + cs3 * cd + sn3 * sd )
    if( data[6] != 0.0 ):
        cd  = numpy.cos( data[7] )
        sd  = numpy.sin( data[7] )
        dph += data[6] * 4.0 * ( cs4 * sd - sn4 * cd )
        mol.func += data[6] * ( 1.0 + cs4 * cd + sn4 * sd )
    if( data[8] != 0.0 ):
        cd  = numpy.cos( data[9] )
        sd  = numpy.sin( data[9] )
        dph += data[8] * 5.0 * ( cs5 * sd - sn5 * cd )
        mol.func += data[8] * ( 1.0 + cs5 * cd + sn5 * sd )
    if( data[10] != 0.0 ):
        cd  = numpy.cos( data[11] )
        sd  = numpy.sin( data[11] )
        dph += data[10] * 6.0 * ( cs6 * sd - sn6 * cd )
        mol.func += data[10] * ( 1.0 + cs6 * cd + sn6 * sd )
    if( grad ):
        dki = mol.coor[a_k] - mol.coor[a_i]
        dlj = mol.coor[a_l] - mol.coor[a_j]
        dvt = numpy.cross( vt, dkj ) / ( rt2 * rkj )
        dvu = numpy.cross( vu, dkj ) / ( ru2 * rkj )
        mol.grad[a_i] += dph * numpy.cross( dvt, dkj )
        mol.grad[a_j] += dph * ( numpy.cross( dki, dvt ) - numpy.cross( dvu, dlk ) )
        mol.grad[a_k] += dph * ( numpy.cross( dvt, dji ) - numpy.cross( dlj, dvu ) )
        mol.grad[a_l] -= dph * numpy.cross( dvu, dkj )
    ang = qm3.data.R2D * numpy.arccos( cs1 )
    if( sn1 <= 0.0 ):
        ang = -ang
    return( ang )


def f_improper( mol: object, kumb: float, xref: float, a_i: int, a_j: int, a_k: int, a_l: int,
        grad: typing.Optional[bool] = False ) -> float:
    """
    improper = force_constant / 2 * ( angle - reference )^2

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
    rtu = numpy.sqrt( rt2 * ru2 )
    rkj = numpy.linalg.norm( dkj )
    cos = numpy.sum( vt * vu ) / rtu
    cos = min( 1.0, max( -1.0, cos ) )
    sin = numpy.sum( dkj * vtu ) / ( rkj * rtu )
    ang = qm3.data.R2D * numpy.arccos( cos )
    if( sin <= 0.0 ):
        ang = -ang
    if( numpy.fabs( ang + xref ) < numpy.fabs( ang - xref ) ):
        xref = -xref
    dt  = ang - xref
    while( dt >  180.0 ):
        dt -= 360.0
    while( dt < -180.0 ):
        dt += 360.0
    dt /= qm3.data.R2D
    mol.func += 0.5 * kumb * dt * dt
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
    return( ang )


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
        mol.rval.append( f_distance( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.skpL, self.skpB ) )

    def get_grad( self, mol: object ):
        mol.rval.append( f_distance( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.skpL, self.skpB, True ) )


class angle( object ):
    def __init__( self, kumb: float, xref: float, indx: list ):
        self.kumb = kumb
        self.xref = xref / qm3.data.R2D
        self.indx = indx[:]

    def get_func( self, mol: object ):
        mol.rval.append( f_angle( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2] ) )

    def get_grad( self, mol: object ):
        mol.rval.append( f_angle( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], True ) )


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
        mol.rval.append( f_dihedral( mol, self.data, self.indx[0], self.indx[1], self.indx[2], self.indx[3] ) )

    def get_grad( self, mol: object ):
        mol.rval.append( f_dihedral( mol, self.data, self.indx[0], self.indx[1], self.indx[2], self.indx[3], True ) )


class improper( object ):
    def __init__( self, kumb: float, xref: float, indx: list ):
        self.kumb = kumb
        self.xref = xref
        self.indx = indx[:]

    def get_func( self, mol: object ):
        mol.rval.append( f_improper( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], self.indx[3] ) )

    def get_grad( self, mol: object ):
        mol.rval.append( f_improper( mol, self.kumb, self.xref, self.indx[0], self.indx[1], self.indx[2], self.indx[3], True ) )


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
        mol.func += 0.5 * self.kumb * ( vv - self.xref ) * ( vv - self.xref )
        mol.rval.append( vv )

    def get_grad( self, mol: object ):
        dr = numpy.zeros( ( self.size, 3 ), dtype=numpy.float64 )
        rr = numpy.zeros( self.size, dtype=numpy.float64 )
        for i in range( self.size ):
            dr[i] = mol.coor[self.indx[2*i]] - mol.coor[self.indx[2*i+1]]
            rr[i] = numpy.linalg.norm( dr[i] )
        vv = numpy.sum( rr * self.weig )
        df = self.kumb * ( vv - self.xref )
        mol.func += 0.5 * df * ( vv - self.xref )
        for i in range( self.size ):
            tt = self.weig[i] * df / rr[i]
            mol.grad[self.indx[2*i]] += tt * dr[i]
            mol.grad[self.indx[2*i+1]] -= tt * dr[i]
        mol.rval.append( vv )



class tether( object ):
    def __init__( self, mol: object, kumb: float,
            sele: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ) ):
        """
    thether = force_constant / 2 * SUM ( cartesian - reference )^2

    force_constant [kJ/mol.A^2]
    reference [A]
        """
        self.kumb = kumb
        if( sele.sum() > 0 ):
            self.sele = sele.reshape( ( mol.natm, 1 ) ) * 1.0
        else:
            self.sele = numpy.ones( ( mol.natm, 1 ) )
        self.cref = mol.coor.copy()

    def get_func( self, mol: object ):
        dr = ( mol.coor - self.cref ) * self.sele
        mol.func += 0.5 * self.kumb * numpy.sum( dr * dr )

    def get_grad( self, mol: object ):
        dr = ( mol.coor - self.cref ) * self.sele
        mol.func += 0.5 * self.kumb * numpy.sum( dr * dr )
        mol.grad += self.kumb * dr
