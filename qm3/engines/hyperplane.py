import  sys
import  math
import  json
import  pickle
import  types
import  numpy
import  typing
import  qm3.data
import  qm3.utils
import  qm3.utils.hessian
import  qm3.engines.mmres


"""
Hiperplano de transicion en el espacio de distancias de una colvar_s
    Zinovjev & Tunon, J. Chem. Phys. v143, p134111 (2015) [doi:10.1063/1.4931596]

colvar_hts ES una colvar_s ( mismo fichero de pares de atomos, mismas distancias d_c, misma metrica
ponderada por masa ), pero en vez de la coordenada de camino s usa

    q( x ) = ( d( x ) - z )^T M^-1 n,        n^T M^-1 n = 1

con ( z, n, M ) optimizados para maximizar el coeficiente de transmision. Misma interfaz que colvar_s
( value / gradient / indx ), asi que sirve directamente en cvs_umb, PMF, etc.

Unidades ( las del articulo ): d [Ang], m [amu] -> q [amu^1/2 Ang], |dq| adimensional;
K [ps^-2]; gamma_z [ps^-1], gamma_n, gamma_p ( valores numericos del articulo ); kT en amu Ang^2 ps^-2.
Los atomos congelados ( mol.actv ) no cuentan en la metrica; se lee mol.actv en CADA evaluacion.

Flujo:  colvar_hts + initial_guess  ->  bias + optimizer ( UNICA etapa con sesgo cuadratico )
        ->  constrained_verlet ( qm3.actions.hts_dynamics ): dinamica en q = 0 SIN restraint
"""

E_CONV = 0.01     # amu Ang^2 ps^-2  ->  kJ/mol


def kT_int( temperature: float ) -> float:
    return( 0.1 * qm3.data.KB * qm3.data.NA * temperature )


class sample( object ):
    __slots__ = ( "theta", "J", "Mt", "q", "gq", "gnorm" )


class colvar_hts( qm3.engines.mmres.colvar_s ):
    """
    gq   = dq/dx = J^T M^-1 n       ( vector plano de 3*len( indx ) )
    |dq| = sqrt( n^T M^-1 Mt M^-1 n ),   Mt = J W J^T  ( metrica local, eq. 9 )
    """
    def __init__( self, mol: object, str_cnf: str ):
        super().__init__( mol, str_cnf, "", 0.0, True )
        self._m0 = self.mass.copy()                  # 1/m sin tener en cuenta mol.actv
        self._w  = self._act_weights( mol )
        self.z = numpy.zeros( self.ncrd )
        self.set_state( self.z, numpy.ones( self.ncrd ), numpy.eye( self.ncrd ) )

    def get_info( self, mol: object ) -> tuple:
        """ como colvar_s.get_info, pero con la masa efectiva sincronizada con mol.actv y mol.mass ACTUALES
            ( los atomos congelados pesan 0 ): set_active() o un cambio isotopico pueden ocurrir entre etapas """
        if( hasattr( mol, "actv" ) ):
            self._w = self._act_weights( mol )
        if( hasattr( mol, "mass" ) ):          # sustituciones isotopicas posteriores a la construccion
            self._m0 = 1.0 / numpy.repeat( mol.mass[self.indx].ravel(), 3 )
        self.mass = self._m0 * self._w         # sin mol.actv ( arrays de coordenadas ): ultimo actv conocido
        return( super().get_info( mol ) )

    def _act_weights( self, mol: object ) -> numpy.array:
        return( numpy.repeat( mol.actv[self.indx].ravel().astype( numpy.float64 ), 3 ) )

    # ---------------------------------------------------------------- estado
    def set_state( self, z, n, M ):
        self.z = numpy.array( z, dtype = numpy.float64 )
        self.M = numpy.array( M, dtype = numpy.float64 )
        self.Minv = numpy.linalg.inv( self.M )
        n = numpy.array( n, dtype = numpy.float64 )
        self.n = n / math.sqrt( float( n @ self.Minv @ n ) )         # eq. A7
        self.u = self.Minv @ self.n

    def perp( self, a: numpy.array ) -> numpy.array:
        """ componente de a ortogonal a n bajo la metrica M [eq. 10] """
        return( a - float( a @ self.u ) * self.n )

    def norm_M( self, a: numpy.array ) -> float:
        return( math.sqrt( float( a @ self.Minv @ a ) ) )

    # ---------------------------------------------------------------- evaluacion
    def evaluate( self, mol: object ) -> sample:
        s = sample()
        s.theta, s.J, imet = self.get_info( mol )
        s.Mt    = numpy.linalg.inv( imet )
        s.q     = float( ( s.theta - self.z ) @ self.u )
        s.gq    = s.J.T @ self.u
        s.gnorm = math.sqrt( max( float( self.u @ s.Mt @ self.u ), 1.e-300 ) )     # eq. A1
        return( s )

    def inplane( self, mol: object ) -> float:
        """ distancia en el plano d = sqrt( ( theta - z )^T M^-1 ( theta - z ) - q^2 ) [amu^1/2 Ang], la misma que usa
            el muro dmax de bias; sirve para vigilar en la produccion ( q = 0 ) hacia donde se desplaza el sistema """
        s = self.evaluate( mol )
        dl = s.theta - self.z
        return( math.sqrt( max( float( dl @ self.Minv @ dl ) - s.q * s.q, 0.0 ) ) )

    def value( self, mol: object ) -> tuple:
        """ ( q, distancias ) """
        s = self.evaluate( mol )
        return( ( s.q, s.theta ) )

    def gradient( self, mol: object ) -> tuple:
        """ ( q, dq/dx con shape ( len( indx ), 3 ), distancias ): como colvar_s.gradient """
        s = self.evaluate( mol )
        return( ( s.q, s.gq.reshape( ( len( self.indx ), 3 ) ), s.theta ) )

    # ---------------------------------------------------------------- hiperplano inicial
    def initial_guess( self, crd_react: list, crd_prod: list ):
        """ z, n, M iniciales de estructuras de reactivos y productos [eqs. 11-14]; listas de
            moleculas o de arrays de coordenadas ( natm, 3 ) """
        def _t( lst ):
            th, mt = [], []
            for c in lst:
                m = c if hasattr( c, "coor" ) else types.SimpleNamespace( coor = numpy.asarray( c ) )
                t, J, imet = self.get_info( m )
                th.append( t ); mt.append( numpy.linalg.inv( imet ) )
            return( numpy.array( th ), numpy.array( mt ) )
        tr, mr = _t( crd_react )
        tp, mp = _t( crd_prod )
        self.set_state( 0.5 * ( tr.mean( axis = 0 ) + tp.mean( axis = 0 ) ),
                tp.mean( axis = 0 ) - tr.mean( axis = 0 ),
                numpy.concatenate( ( mr, mp ) ).mean( axis = 0 ) )
        self.z0, self.n0 = self.z.copy(), self.n.copy()

    def ts_mode( self, mol: object, hess: numpy.array, sele: typing.Optional[numpy.array] = None,
            project: typing.Optional[bool] = False ) -> dict:
        """
        modo imaginario del Hessiano cartesiano ( ver initial_guess_ts para sele / project ):
        devuelve { "dx": desplazamiento cartesiano ( natm, 3 ) por unidad de paso ponderado por masa, "freq": cm^-1,
                   "nneg": numero de autovalores negativos, "w", "e", "sele": autovalores ( kJ/mol/amu/Ang^2 ),
                   autovectores ponderados por masa y atomos del Hessiano }
        """
        if( sele is None ):
            sele = numpy.flatnonzero( mol.actv.ravel() )
        else:
            sele = numpy.asarray( sele )
            sele = numpy.flatnonzero( sele.ravel() ) if sele.dtype == numpy.bool_ else numpy.sort( sele.ravel() )
            if( project ):
                raise ValueError( "colvar_hts.initial_guess_ts: project=True requiere sele = mol.actv" )
        if( hess.shape[0] != 3 * len( sele ) ):
            raise ValueError( "colvar_hts.initial_guess_ts: el Hessiano es %d x %d y sele tiene %d atomos"%( hess.shape[0], hess.shape[1], len( sele ) ) )
        m  = numpy.repeat( mol.mass[sele].ravel(), 3 )
        Hw = hess / numpy.sqrt( numpy.outer( m, m ) )
        if( project ):
            Hw = qm3.utils.hessian.project_RT( Hw, qm3.utils.RT_modes( mol ) )
        w, e = numpy.linalg.eigh( Hw )
        nneg = int( numpy.sum( w < - 1.e-6 * max( abs( w ).max(), 1.e-30 ) ) )
        if( nneg == 0 ):
            raise ValueError( "colvar_hts.initial_guess_ts: el Hessiano no tiene ningun autovalor negativo ( no es un TS )" )
        dx = numpy.zeros( ( mol.natm, 3 ) )
        dx[sele] = ( e[:,0] / numpy.sqrt( m ) ).reshape( ( len( sele ), 3 ) )      # desplazamiento cartesiano del modo
        return( { "dx": dx, "freq": - math.sqrt( abs( w[0] ) ) * 1.0e11 / ( 2. * math.pi * qm3.data.C ), "nneg": nneg,
                  "w": w, "e": e, "sele": sele } )

    def k_min( self, mol: object, md: dict ) -> float:
        """
        K minima ( ps^-2 ) para que V + K q^2 / 2 sea localmente estable en el TS, es decir para que el sesgo pueda
        confinar al walker en el plano frente a la curvatura negativa de la barrera:

            K_min = 1 / ( - g^T H^-1 g ),     g = dq/dx ponderado por masa,  H = Hessiano ponderado por masa

        ( si g es paralelo al modo imaginario, K_min = omega_imag^2 ). Usa el Hessiano de md = ts_mode( ... ), con el
        entorno congelado, asi que es una estimacion: con el entorno movil la curvatura efectiva suele ser mayor.
        Con K <= K_min los walkers escapan del plano ( el ensemble sesgado no esta confinado ) y el optimizador
        no es fiable; usar K >~ 1.5 K_min. Devuelve inf si ningun K estabiliza.
        """
        sele, w, e = md["sele"], md["w"], md["e"]
        gfull = numpy.zeros( ( mol.natm, 3 ) )
        gfull[self.indx] = self.evaluate( mol ).gq.reshape( ( len( self.indx ), 3 ) )
        g = ( gfull[sele] / numpy.sqrt( mol.mass[sele] ) ).ravel()
        lam = 100.0 * w                                       # ps^-2
        use = numpy.abs( lam ) > 1.0e-6 * numpy.abs( lam ).max()
        S = float( numpy.sum( ( e.T @ g )[use]**2 / lam[use] ) )
        return( 1.0 / ( - S ) if S < 0.0 else float( "inf" ) )

    def initial_guess_ts( self, mol: object, hess: numpy.array,
            sele: typing.Optional[numpy.array] = None,
            project: typing.Optional[bool] = False,
            react: typing.Optional[object] = None ) -> dict:
        """
        z, n, M iniciales desde una estructura de TS localizada en la PES ( mol en el TS ) y su Hessiano
        cartesiano ( solo atomos activos, kJ/mol/Ang^2, p.ej. qm3.utils.hessian.numerical ):
            z = d( TS ),  M = Mt( TS ),  n ~ J . ( M^-1/2 e )   con e el modo imaginario ponderado por masa
        Diagonaliza el Hessiano aqui mismo ( no usa hessian.frequencies ).

        mol.actv debe ser el conjunto de atomos MOVILES de la dinamica ( con el que se calcula la metrica ).
        sele:    atomos a los que corresponde el Hessiano ( mascara booleana natm o indices ), p.ej. solo la
                 region QM si hess se calculo con set_active( QM ). Por defecto, mol.actv. NO llames a esta
                 funcion con mol.actv = solo QM si alguna distancia une atomos que quedarian sin activar:
                 su metrica seria singular.

        project: elimina traslaciones/rotaciones. Solo True para una molecula aislada: en QM/MM con el entorno
                 congelado el subsistema activo NO es invariante y proyectar seria incorrecto
        react:   estructura ( molecula o array ) del lado de reactivos, solo para fijar el signo de q
                 ( q < 0 hacia reactivos ). Sin ella el signo es el de la fase arbitraria del autovector.
        devuelve { "overlap": fraccion del modo imaginario expresable con estas distancias ( 0-1 ),
                   "freq": cm^-1 ( negativa ), "nneg": numero de autovalores negativos,
                   "kmin": K minima ( ps^-2 ) para confinar al walker en el plano, ver k_min }
        """
        md = self.ts_mode( mol, hess, sele, project )
        dx, nneg = md["dx"], md["nneg"]
        th, J, imet = self.get_info( mol )
        r  = J @ dx[self.indx].ravel()                                              # d theta por unidad de paso ponderado
        Mt = numpy.linalg.inv( imet )
        ov = math.sqrt( max( float( r @ imet @ r ), 0.0 ) )
        if( ov < 1.e-3 ):
            raise ValueError( "colvar_hts.initial_guess_ts: el modo imaginario no se proyecta en estas distancias ( overlap = %.1e )"%( ov ) )
        if( react is not None ):
            t0, _, _ = self.get_info( react if hasattr( react, "coor" ) else types.SimpleNamespace( coor = numpy.asarray( react ) ) )
            if( float( ( t0 - th ) @ imet @ r ) > 0.0 ):
                r = - r
        self.set_state( th, r, Mt )
        self.z0, self.n0 = self.z.copy(), self.n.copy()
        return( { "overlap": ov, "freq": md["freq"], "nneg": nneg, "kmin": self.k_min( mol, md ) } )

    # ---------------------------------------------------------------- E/S
    def save( self, fname: str ):
        with open( fname, "wt" ) as f:
            json.dump( { "atom": self.atom.tolist(), "z": self.z.tolist(),
                         "n": self.n.tolist(), "M": self.M.tolist() }, f, indent = 1 )

    def load( self, fname: str ):
        """ el fichero de pares de la colvar debe ser el mismo que en la optimizacion """
        with open( fname, "rt" ) as f:
            d = json.load( f )
        if( d["atom"] != self.atom.tolist() ):
            raise ValueError( "colvar_hts.load: los pares de atomos no coinciden" )
        self.set_state( d["z"], d["n"], d["M"] )


# =================================================================================================
# Optimizacion del hiperplano ( UNICA etapa que usa el sesgo cuadratico del articulo )
# =================================================================================================

class bias( object ):
    """
    V_HTS = V + K / 2 q^2                              [eq. 3]
    + muro en el plano, opcional ( dmax ):  K / 2 ( d - dmax )^2  si d > dmax   [eq. 20]
      con d^2 = ( theta - z )^T M^-1 ( theta - z ) - q^2
    K [ps^-2]. Guarda el ultimo 'sample' en self.sample ( lo consume optimizer.collect )
    """
    def __init__( self, hts: colvar_hts, kumb: float, dmax: typing.Optional[float] = None ):
        self.hts, self.kumb, self.dmax = hts, kumb, dmax
        self.sample = None
        self.ncall = self.nwall = 0            # pasos totales y pasos con el muro activo ( d > dmax )
        self.dsum = self.dmax_seen = 0.0

    def wall_report( self ) -> tuple:
        """ ( fraccion de pasos con el muro activo, <d>, max d ) desde la ultima llamada a reset_wall();
            si la fraccion es grande ( > ~5% ) dmax esta sesgando el plano optimizado """
        n = max( self.ncall, 1 )
        return( ( self.nwall / n, self.dsum / n, self.dmax_seen ) )

    def reset_wall( self ):
        self.ncall = self.nwall = 0
        self.dsum = self.dmax_seen = 0.0

    def _eval( self, mol: object ) -> tuple:
        h = self.hts
        s = h.evaluate( mol )
        e = 0.5 * self.kumb * s.q * s.q
        g = self.kumb * s.q * s.gq
        dl = s.theta - h.z
        d  = math.sqrt( max( float( dl @ h.Minv @ dl ) - s.q * s.q, 0.0 ) )
        if( self.dmax is not None and d > self.dmax ):
            e += 0.5 * self.kumb * ( d - self.dmax )**2
            g = g + s.J.T @ ( self.kumb * ( d - self.dmax ) * ( h.Minv @ dl - s.q * h.u ) / d )
        self.sample = s
        return( ( E_CONV * e, E_CONV * g, s.q, d ) )

    def get_func( self, mol: object ) -> tuple:
        e, g, q, d = self._eval( mol )
        mol.func += e
        return( ( e, q, d ) )

    def get_grad( self, mol: object ) -> tuple:
        e, g, q, d = self._eval( mol )
        mol.func += e
        mol.grad[self.hts.indx] += g.reshape( ( len( self.hts.indx ), 3 ) )
        self.ncall += 1
        self.dsum += d
        self.dmax_seen = max( self.dmax_seen, d )
        if( self.dmax is not None and d > self.dmax ):
            self.nwall += 1
        return( ( e, q, d ) )


class optimizer( object ):
    """
    [eqs. 15, 16, 18, 19, 24, 25]. En cada paso de MD de cada walker:   opt.collect( bias.sample )
    cada 'nsync' pasos ( igual en todos los walkers ):                    opt.update( nsync * step_size )
    Con 'opar' ( client_fsi / client_mpi ) hay un walker por proceso y todos conservan el mismo
    hiperplano ( como metadyn.run_mw ); sin opar se promedia lo recogido localmente.
    Todos los walkers deben partir del MISMO hiperplano inicial, mitad en reactivos, mitad en productos.
    """
    def __init__( self, hts: colvar_hts, kumb: float,
            temperature: typing.Optional[float] = 300.0,
            gamma_z: typing.Optional[float] = 2000.0,
            gamma_n: typing.Optional[float] = 2000.0,
            gamma_p: typing.Optional[float] = 1000.0,
            opar: typing.Optional[object] = None,
            log_file: typing.Optional[typing.IO] = None,
            walker_log: typing.Optional[typing.IO] = None ):
        self.hts, self.kumb, self.kT = hts, kumb, kT_int( temperature )
        self.wlog = walker_log
        self.gz, self.gn, self.gp = gamma_z, gamma_n, gamma_p
        self.opar, self.log = opar, log_file
        self.time = self.tM = 0.0
        self.hist = []
        self.z0, self.n0 = hts.z.copy(), hts.n.copy()
        self.store, self.stride, self.sth, self.smt = False, 1, [], []
        self._reset()

    def _reset( self ):
        D = self.hts.ncrd
        self.cnt = 0
        self.az, self.bn, self.pz = numpy.zeros( D ), numpy.zeros( D ), numpy.zeros( D )
        self.mt = numpy.zeros( ( D, D ) )
        self.qs = self.ds = 0.0                 # diagnosticos LOCALES ( de este proceso ) desde el ultimo update
        self.qx = self.dx = 0.0
        self.ths = numpy.zeros( D )

    def start_storing( self, stride: typing.Optional[int] = 10 ):
        """ guarda ( d, Mt ) cada 'stride' pasos para importance(); llamar tras la convergencia """
        self.store, self.stride, self.sth, self.smt = True, stride, [], []

    def collect( self, s: sample ):
        h  = self.hts
        dp = h.perp( s.theta - h.z )
        kq = self.kumb * s.q
        self.az += s.gnorm * kq * h.n                                                  # eq. 15
        self.bn += s.gnorm * kq * dp - self.kT / s.gnorm * h.perp( s.Mt @ h.u )        # eq. 16
        self.pz += dp                                                                  # eq. 19
        self.mt += s.Mt
        self.cnt += 1
        if( self.wlog is not None ):
            dl = s.theta - h.z
            d  = math.sqrt( max( float( dl @ h.Minv @ dl ) - s.q * s.q, 0.0 ) )
            self.qs += s.q; self.qx = max( self.qx, abs( s.q ) )
            self.ds += d;   self.dx = max( self.dx, d )
            self.ths += s.theta
        if( self.store and self.cnt % self.stride == 0 ):
            self.sth.append( s.theta.copy() )
            self.smt.append( s.Mt.copy() )

    def _reduce( self ) -> tuple:
        D = self.hts.ncrd
        vec = numpy.concatenate( ( [ self.cnt ], self.az, self.bn, self.pz, self.mt.ravel() ) )
        op = self.opar
        if( op is not None ):
            if( op.node == 0 ):
                for who in range( 1, op.ncpu ):
                    vec = vec + numpy.array( op.recv_r8( who, len( vec ) ) )
                for who in range( 1, op.ncpu ):
                    op.send_r8( who, vec.tolist() )
            else:
                op.send_r8( 0, vec.tolist() )
                vec = numpy.array( op.recv_r8( 0, len( vec ) ) )
        if( vec[0] < 1 ):
            raise RuntimeError( "optimizer.update: nothing collected" )
        vec = vec[1:] / vec[0]
        return( ( vec[:D], vec[D:2*D], vec[2*D:3*D], vec[3*D:].reshape( ( D, D ) ) ) )

    def update( self, dt: float ):
        """ avanza el hiperplano dt [ps] """
        h = self.hts
        if( self.wlog is not None and self.cnt > 0 ):          # promedios locales ANTES de la reduccion entre walkers
            D = h.ncrd
            row = [ self.time + dt, self.qs / self.cnt, self.qx, self.ds / self.cnt, self.dx,
                    numpy.linalg.norm( self.az / self.cnt ), numpy.linalg.norm( self.bn / self.cnt ) ]
            row += list( self.ths / self.cnt ) + list( h.z ) + list( h.n )
            if( self.time == 0.0 ):
                self.wlog.write( "#K %.8g kT %.8g\n"%( self.kumb, self.kT ) )
                self.wlog.write( "#time  <q>  max|q|  <d>  max_d  |az|  |bn|  " + "  ".join( [ "theta%d"%( i ) for i in range( D ) ] +
                    [ "z%d"%( i ) for i in range( D ) ] + [ "n%d"%( i ) for i in range( D ) ] ) + "\n" )
            self.wlog.write( " ".join( "%.6g"%( x ) for x in row ) + "\n" )
            self.wlog.flush()
        az, bn, pz, mt = self._reduce()
        z = h.z + dt * ( - az / self.gz + pz / self.gp )
        n = h.n + dt * bn / self.gn
        M = ( self.tM * h.M + dt * mt ) / ( self.tM + dt )             # eq. 18
        self.tM += dt
        self.time += dt
        h.set_state( z, n, M )
        self._reset()
        cz = h.norm_M( h.z - self.z0 )                                  # eq. 22
        n0 = self.n0 / h.norm_M( self.n0 )
        cn = math.degrees( math.acos( min( 1.0, max( -1.0, float( h.n @ h.Minv @ n0 ) ) ) ) )   # eq. 23
        self.hist.append( ( self.time, h.z.copy(), h.n.copy(), cz, cn ) )
        if( self.log is not None ):
            self.log.write( "%16.6lf%16.8lf%16.8lf\n"%( self.time, cz, cn ) )
            self.log.flush()

    def save_history( self, fname: str ):
        """ guarda opt.hist ( time, z, n, Cz, Cn ) para promediar a posteriori con final_from_files """
        with open( fname, "wb" ) as f:
            pickle.dump( self.hist, f )

    def final( self, t_conv: float ) -> tuple:
        """ promedios de z y n para t >= t_conv [eqs. 24, 25]; los aplica a hts """
        sel = [ i for i in self.hist if i[0] >= t_conv ]
        if( not sel ):
            raise ValueError( "optimizer.final: no data after t_conv" )
        self.hts.set_state( numpy.mean( [ i[1] for i in sel ], axis = 0 ),
                numpy.mean( [ i[2] for i in sel ], axis = 0 ), self.hts.M )
        return( ( self.hts.z.copy(), self.hts.n.copy() ) )

    def importance( self, kumb: typing.Optional[float] = None, return_ess: typing.Optional[bool] = False ) -> tuple:
        return( importance_analysis( self.hts, numpy.array( self.sth ), numpy.array( self.smt ),
                self.kumb if kumb is None else kumb, self.kT, return_ess ) )


def importance_analysis( hts: colvar_hts, theta: numpy.array, Mt: numpy.array, kumb: float, kT: float,
        return_ess: typing.Optional[bool] = False ) -> tuple:
    """
    kappa( q_-i ) / kappa( q ) al quitar cada distancia [eqs. 26 y B6]; ~1 irrelevante, << 1 esencial.
    theta ( S, D ), Mt ( S, D, D ): muestras CON el sesgo ( bias ) de constante kumb tras converger.
    devuelve ( eq26, eqB6 ); con return_ess = True, ( eq26, eqB6, ess ) donde ess[i] es el tamano efectivo de muestra
    de los pesos exp( beta K / 2 ( q^2 - q_-i^2 ) ) de la CV i ( ess << numero de muestras: solape pobre, valor NO fiable;
    el apendice B del articulo recomienda K pequena para este analisis )
    """
    D  = hts.ncrd
    dl = theta - hts.z
    q  = dl @ hts.u
    gn = numpy.sqrt( numpy.einsum( "i,sij,j->s", hts.u, Mt, hts.u ) )
    r26, rB6, ess = numpy.zeros( D ), numpy.zeros( D ), numpy.zeros( D )
    for i in range( D ):
        v = hts.u.copy(); v[i] = 0.0
        ui = v / math.sqrt( float( v @ hts.M @ v ) )                    # M^-1 n_-i  [B3]
        qi = dl @ ui
        gi = numpy.sqrt( numpy.einsum( "i,sij,j->s", ui, Mt, ui ) )
        b  = 0.5 * kumb / kT * ( q * q - qi * qi )
        bm = b.max()
        eb = numpy.exp( b - bm )
        r26[i] = gn.mean() * math.exp( - bm ) / ( gi * eb ).mean()
        rB6[i] = math.exp( - bm ) / eb.mean()
        ess[i] = float( eb.sum()**2 / numpy.sum( eb * eb ) )
    return( ( r26, rB6, ess ) if return_ess else ( r26, rB6 ) )


def final_from_files( hist_file: str, json_last: str, t_conv: float, json_out: str,
        t_end: typing.Optional[float] = None ) -> tuple:
    """
    promedia z y n de la historia guardada ( optimizer.save_history ) en la ventana [ t_conv, t_end ] y escribe el
    .json del hiperplano final, sin necesitar construir el sistema. json_last: cualquier .json guardado con hts.save
    durante la optimizacion ( aporta M y los pares de atomos ). Permite elegir la ventana estacionaria DESPUES de
    ver Cz / Cn ( no se debe promediar sobre una deriva ).
    """
    with open( hist_file, "rb" ) as f:
        hist = pickle.load( f )
    sel = [ h for h in hist if h[0] >= t_conv and ( t_end is None or h[0] <= t_end ) ]
    if( not sel ):
        raise ValueError( "final_from_files: no hay datos en la ventana pedida" )
    with open( json_last, "rt" ) as f:
        d = json.load( f )
    M = numpy.array( d["M"], dtype = numpy.float64 )
    z = numpy.mean( [ h[1] for h in sel ], axis = 0 )
    n = numpy.mean( [ h[2] for h in sel ], axis = 0 )
    n = n / math.sqrt( float( n @ numpy.linalg.inv( M ) @ n ) )
    with open( json_out, "wt" ) as f:
        json.dump( { "atom": d["atom"], "z": z.tolist(), "n": n.tolist(), "M": M.tolist() }, f, indent = 1 )
    return( ( z, n, len( sel ), sel[0][0], sel[-1][0] ) )
