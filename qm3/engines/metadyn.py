import  math
import  numpy
import  typing
import  qm3.data
import  qm3.engines.mmres


# path/string collective variable s
# J. Chem. Phys. v136, p224101 (2012) [doi:10.1063/1.4723780]
# J. Comput. Chem. v35, p1672 (2014) [doi:10.1002/jcc.23673]
# WIREs Comput. Mol. Sci. v8 (2018) [doi:10.1002/wcms.1329]
# ------------------------------------------------------------------------
# metadynamics
# Proc. Natl. Acad. Sci. v99, p12562 (2002) [doi:10.1073/pnas.202628299]
# time-averaged/metadynamics with arbitrary CVs
# J. Phys. Chem. Lett. v6, p375 (2015) [doi:10.1021/jz5027393]
# ------------------------------------------------------------------------
# well-tempered metadynamics
# Phys. Rev. Lett. v100, p020603 (2008) [doi:10.1103/PhysRevLett.100.020603]


class run( object ):
    """
    Metadynamics on a collective variable s, given as a ready-to-use
    qm3.engines.mmres.colvar_s object (the string/path CV of Branduardi et al.;
    the CV knows how to compute s and ds/dR, this class only builds the bias).

    A bias potential composed of Gaussians,
        V(s,t) = SUM_{t' < t} w(t') exp( -( s - s(t') )^2 / 2 sigm^2 ),
    is added to the Hamiltonian. The flavour is selected by the arguments:

    * flooding (default): delta, temperature and wmin are NOT given.
        Classical metadynamics: constant hill height w = wght. V keeps growing
        and fluctuates around F(s) ~ - V(s,t) + const.

    * well-tempered: delta AND temperature are given (wmin is an optional
        extra). The height of every new hill is re-scaled
            w(t) = wght * exp( - V( <s>, t ) / ( kB delta ) )
        so that the bias converges and the sampling follows the modified free
        energy; F(s) = - ( T + delta ) / delta * V(s,t->inf),
        gamma = ( T + delta ) / T (delta = 1500 K @ 300 K -> gamma = 6).
        wmin (0 < wmin < wght) sets a lower bound w(t) = max( w(t), wmin ).
        Use it with care: once V > kB delta ln( wght / wmin ) the run is plain
        metadynamics with constant height wmin, the bias does not converge any
        more and get_pmf is NOT valid. Without wmin it is pure WT-Meta.

    Giving only part of the set (delta without temperature, temperature
    without delta, or wmin alone) is an error, to avoid silently falling back
    to flooding.

    Usage (engine interface, see qm3.actions.dynamics):
        cv = qm3.engines.mmres.colvar_s( mol, "pmf_s.cnf", "pmf_s.str", delz = 0.2117 )
        # flooding
        mol.engines["meta"] = qm3.engines.metadyn.run( cv, sigm = 0.025, wght = 1.0,
                                nstp = 100, wall = [ 250.0, 0.2, 6.0 ], name = "pmf_s.meta" )
        # well-tempered
        mol.engines["meta"] = qm3.engines.metadyn.run( cv, sigm = 0.025, wght = 1.0,
                                nstp = 100, delta = 1500.0, temperature = 300.0,
                                name = "pmf_s.wtm" )
        qm3.actions.dynamics.langevin_verlet( mol, ... )

    The returned value is stored in mol.rval["meta"] = ( V_bias, s, internal-coords ).

    Deposition is a side effect of get_grad(): hills are added every nstp calls.
    Set the attribute "deposit" to False to evaluate the bias without growing it
    (minimizations, line searches, extra gradient calls, ...).

    Parameters
    -------------------------------------
    colvar  : qm3.engines.mmres.colvar_s with the string already defined
        (>= 2 references and delz > 0, i.e. after define() or loading str_crd
        together with delz)
    sigm    : width of the Gaussian hills [units of s], use ~ the rms fluctuation
        of s within a sampling window
    wght    : (initial) height of the Gaussian hills [kJ/mol]; in WT-Meta keep it
        small, 1-2 k BT is appropriate
    nstp    : deposition stride [MD steps]; the time-averaged <s> over the last
        nstp steps is used as the center of the new hill
    delta   : (WT) bias temperature delta-T [K]
    temperature : (WT) MD temperature [K]
    wmin    : (WT, optional) lower bound to the hill height [kJ/mol]
    wall    : [ kumb, llim, ulim ] optional harmonic walls on s [units of s, i.e.
        the same units as delz, NOT normalized]: kumb [kJ/mol/(s units)^2],
        llim/ulim [s units]. The wall is zero inside [llim,ulim] and continuous
        (V and dV/ds) at the limits. Gaussians are deposited only while
        llim <= s <= ulim
    name    : optional log file of the deposited hills (opened in append mode
        if str_hills is given, i.e. on restart)
    str_hills : optional file of previously deposited hills for restart
        (two columns: s_center height [kJ/mol], see save_hills)
    str_save  : optional file where the hills are saved (save_hills)
    nsav    : number of DEPOSITIONS (not MD steps, that is nstp) between saves
        of str_save. The hills are also saved when the object is closed, so the
        last ones are never lost. Only used if str_save is given

    s units:
    -------------------------------------
    kumb              kJ / mol / (s units)^2
    sigm/delz/s       Ang (Ang amu^0.5 if the colvar is mass-weighted)
    wght/wmin/V       kJ / mol
    """
    def __init__( self, colvar: qm3.engines.mmres.colvar_s,
            sigm: float, wght: float,
            nstp: typing.Optional[int] = 100,
            delta: typing.Optional[float] = None,
            temperature: typing.Optional[float] = None,
            wmin: typing.Optional[float] = None,
            wall: typing.Optional[list] = [],
            name: typing.Optional[str] = None,
            str_hills: typing.Optional[str] = "",
            str_save: typing.Optional[str] = "",
            nsav: typing.Optional[int] = 100 ):
        # fdsc first, so that __del__/close are always safe
        self.fdsc = None
        self.sfil = ""
        self._nsvd = 0
        if( not isinstance( colvar, qm3.engines.mmres.colvar_s ) ):
            raise TypeError( "metadyn.run: colvar should be a qm3.engines.mmres.colvar_s object" )
        if( colvar.nwin < 2 ):
            raise ValueError( "metadyn.run: at least two references are needed in the colvar" )
        if( colvar.delz <= 0.0 ):
            raise ValueError( "metadyn.run: delz is undefined in the colvar" )
        if( sigm <= 0.0 ):
            raise ValueError( "metadyn.run: sigm should be positive" )
        if( wght <= 0.0 ):
            raise ValueError( "metadyn.run: wght (hill height) should be positive" )
        if( nstp < 1 ):
            raise ValueError( "metadyn.run: nstp should be a positive integer" )
        if( len( wall ) != 0 ):
            if( len( wall ) != 3 ):
                raise ValueError( "metadyn.run: wall should be [ kumb, llim, ulim ]" )
            if( wall[1] >= wall[2] ):
                raise ValueError( "metadyn.run: wrong wall definition (llim >= ulim)" )
        # flavour: well-tempered needs delta AND temperature; wmin is optional
        self.wt = ( delta != None and temperature != None )
        if( not self.wt and ( delta != None or temperature != None or wmin != None ) ):
            raise ValueError( "metadyn.run: well-tempered needs both delta and temperature (wmin is optional); " +
                    "leave delta, temperature and wmin unset for flooding" )
        self.delta = delta
        self.temp  = temperature
        self.wmin  = wmin
        if( self.wt ):
            if( delta <= 0.0 ):
                raise ValueError( "metadyn.run: delta (bias temperature) should be positive" )
            if( temperature <= 0.0 ):
                raise ValueError( "metadyn.run: temperature should be positive" )
            if( wmin != None and ( wmin <= 0.0 or wmin >= wght ) ):
                raise ValueError( "metadyn.run: wmin should be in ( 0, wght )" )
            # kB [kJ/mol/K] (qm3.data.KB in J/K)
            self.kbdt  = 0.001 * delta * qm3.data.KB * qm3.data.NA
            self.kbt   = 0.001 * temperature * qm3.data.KB * qm3.data.NA
            # gamma = ( T + delta ) / T
            self.gamma = ( self.kbt + self.kbdt ) / self.kbt
        self.cv    = colvar
        self.sigm  = sigm
        self.wght  = wght
        self.nstp  = int( nstp )
        self.wall  = list( wall )
        self.deposit = True
        # hills
        self.hill_s = []
        self.hill_w = []
        self._hs    = numpy.array( [], dtype=numpy.float64 )
        self._hw    = numpy.array( [], dtype=numpy.float64 )
        # deposition book-keeping
        self.step = 0
        self.nall = 0
        self.accu = 0.0
        self.sfil = str_save
        self.nsav = max( 1, int( nsav ) )
        # restart
        restart = ( str_hills != "" )
        if( restart ):
            self.load_hills( str_hills )
        self._nsvd = self.nhill
        # log file
        if( name != None ):
            self.fdsc = open( name, "at" if restart else "wt" )
            if( not restart ):
                self.fdsc.write( self._header() )
                self.fdsc.write( "#%20s%20s%20s%20s%20s\n"%( "step", "s_center", "height", "height_raw", "V_bias" ) )
                self.fdsc.flush()


    def _header( self ) -> str:
        if( self.wt ):
            return( "# Well-Tempered Metadynamics of the string CV s (sigm %.5lf wght %.5lf delta %.2f T %.2f nstp %d wmin %.5lf)\n"%
                    ( self.sigm, self.wght, self.delta, self.temp, self.nstp,
                      0.0 if self.wmin == None else self.wmin ) )
        return( "# Metadynamics (flooding) of the string CV s (sigm %.5lf wght %.5lf nstp %d)\n"%
                ( self.sigm, self.wght, self.nstp ) )


    def __del__( self ):
        try:
            self.close()
        except:
            pass


    def close( self ):
        # flush the last hills (the ones after the last periodic save)
        if( self.sfil != "" and self.nhill > self._nsvd ):
            self.save_hills( self.sfil )
        if( getattr( self, "fdsc", None ) != None ):
            self.fdsc.close()
            self.fdsc = None


    @property
    def nhill( self ) -> int:
        return( len( self.hill_s ) )


    def get_bias( self, sval: typing.Union[ float, numpy.array ],
            grad: typing.Optional[bool] = False ) -> tuple:
        """
        bias potential V(s) [kJ/mol] built from the hills deposited so far
        (and dV/ds [kJ/mol per s unit] if grad); accepts a scalar or an array
        """
        if( self.nhill == 0 ):
            if( hasattr( sval, "__len__" ) ):
                zro = numpy.zeros( len( sval ), dtype=numpy.float64 )
                return( ( zro, zro.copy() ) if grad else zro )
            return( ( 0.0, 0.0 ) if grad else 0.0 )
        shp  = numpy.shape( sval )
        arr  = numpy.reshape( numpy.array( sval, dtype=numpy.float64 ), ( -1, 1 ) )
        dsv  = ( arr - self._hs ) / self.sigm
        g    = self._hw * numpy.exp( -0.5 * dsv * dsv )
        vv   = g.sum( axis = 1 ).reshape( shp )
        if( grad ):
            return( ( vv, -( dsv / self.sigm * g ).sum( axis = 1 ).reshape( shp ) ) )
        return( vv )


    def get_wall( self, sval: float ) -> tuple:
        """
        harmonic walls on s: returns ( V_wall, dV_wall/ds ); zero inside
        [ llim, ulim ], continuous in V and dV/ds at the limits
        """
        if( len( self.wall ) != 3 ):
            return( ( 0.0, 0.0 ) )
        if( sval < self.wall[1] ):
            dlt = sval - self.wall[1]
        elif( sval > self.wall[2] ):
            dlt = sval - self.wall[2]
        else:
            return( ( 0.0, 0.0 ) )
        return( ( 0.5 * self.wall[0] * dlt * dlt, self.wall[0] * dlt ) )


    def _height( self, vbar: float ) -> tuple:
        """
        ( height, raw height ) of the next hill given the bias vbar at its center:
        constant for flooding, re-scaled for well-tempered (optionally bounded
        from below by wmin)
        """
        if( not self.wt ):
            return( ( self.wght, self.wght ) )
        wraw = self.wght * math.exp( - max( vbar, 0.0 ) / self.kbdt )
        wval = max( wraw, self.wmin ) if self.wmin != None else wraw
        return( ( wval, wraw ) )


    def _deposit( self, sval: float ):
        """
        add a Gaussian centered at sval
        """
        vbar = float( self.get_bias( sval ) )
        wval, wraw = self._height( vbar )
        self.hill_s.append( sval )
        self.hill_w.append( wval )
        self._hs = numpy.append( self._hs, sval )
        self._hw = numpy.append( self._hw, wval )
        if( self.fdsc != None ):
            self.fdsc.write( "%20.6lf%20.10lf%20.10lf%20.10lf%20.10lf\n"%
                             ( float( self.nall ), sval, wval, wraw, vbar ) )
            self.fdsc.flush()
        if( self.sfil != "" and self.nhill % self.nsav == 0 ):
            self.save_hills( self.sfil )
        return( wval )


    def get_func( self, mol: object ) -> tuple:
        """
        engine interface: bias potential at the current value of s (no deposition),
        returns ( V_bias, s, internal-coords )
        """
        cval, ccrd = self.cv.value( mol )
        vtot = float( self.get_bias( cval ) ) + self.get_wall( cval )[0]
        mol.func += vtot
        return( ( vtot, cval, ccrd ) )


    def get_grad( self, mol: object ) -> tuple:
        """
        engine interface: bias potential and gradient, and deposition of the
        time-averaged s every nstp calls (if self.deposit),
        returns ( V_bias, s, internal-coords )
        """
        cval, dsdr, ccrd = self.cv.gradient( mol )
        # bias potential from the hills deposited so far + harmonic walls on s
        vbia, dvds = self.get_bias( cval, grad = True )
        vwal, dwds = self.get_wall( cval )
        vtot = float( vbia ) + vwal
        mol.func += vtot
        mol.grad[self.cv.indx,:] += ( float( dvds ) + dwds ) * dsdr
        # deposition of a new hill every nstp steps (mean over the interval),
        # only when inside the walls
        if( self.deposit ):
            self._collect( cval )
        return( ( vtot, cval, ccrd ) )


    def _collect( self, cval: float ):
        """
        deposition book-keeping for one gradient call (overridden in run_mw)
        """
        if( len( self.wall ) == 3 ):
            depo = cval >= self.wall[1] and cval <= self.wall[2]
        else:
            depo = True
        if( depo ):
            self.step += 1
            self.nall += 1
            self.accu += cval
            if( self.step % self.nstp == 0 ):
                self._deposit( self.accu / float( self.nstp ) )
                self.accu = 0.0


    def get_pmf( self, grid: numpy.array,
            shift: typing.Optional[bool] = True ) -> numpy.array:
        """
        free-energy estimate along s from the bias potential, on the given grid
        (approximate, Gaussian-convolved, only meaningful once the bias has
        filled the wells; shifted to the minimum if shift):
            flooding:        F(s) = - V(s,t)
            well-tempered:   F(s) = - ( T + delta ) / delta * V(s,t)
                             (valid only in the pure WT regime, i.e. no wmin)
        """
        v    = numpy.array( self.get_bias( grid, grad = False ), dtype=numpy.float64 )
        if( self.wt ):
            pmf = - ( self.kbt + self.kbdt ) / self.kbdt * v
        else:
            pmf = - v
        if( shift ):
            pmf -= pmf.min()
        return( pmf )


    def save_hills( self, str_hills: str ):
        if( self.nhill > 0 ):
            numpy.savetxt( str_hills, numpy.column_stack( ( self._hs, self._hw ) ),
                           header = "%12s%12s"%( "s_center", "height" ), comments = "# " )
            if( str_hills == self.sfil ):
                self._nsvd = self.nhill


    def load_hills( self, str_hills: str ):
        hls = numpy.loadtxt( str_hills, dtype=numpy.float64 )
        if( hls.size > 0 ):
            hls.shape = ( -1, 2 )
            self.hill_s.extend( hls[:,0].tolist() )
            self.hill_w.extend( hls[:,1].tolist() )
            self._hs = numpy.array( self.hill_s, dtype=numpy.float64 )
            self._hw = numpy.array( self.hill_w, dtype=numpy.float64 )



class run_mw( run ):
    """
    Multiple-walkers version of run (flooding or well-tempered, selected by the
    same arguments): N processes sample at the same time, every walker feels the
    bias built from the hills of ALL the walkers.

    Every walker is a full simulation (own molecule, engines and dynamics) that
    creates its own run_mw object with the SAME parameters and the parallel
    object of qm3.utils.parallel (client_fsi or client_mpi); all the ncpu
    processes are walkers (unlike neb, there is no dedicated master).

    Synchronization: every nstp gradient calls (counted in ALL the walkers
    whether or not they are inside the walls) the walkers meet at a
    deposition round arbitrated by walker 0:
        1. every walker sends its time-averaged s over the interval (only
           the steps inside the walls are averaged; no hill if there is none)
        2. walker 0 collects them and sends the whole list back
        3. every walker computes the height of the hills with the bias as it was
           BEFORE the round (so that the result does not depend on the order
           of the walkers) and appends them in walker order
    Therefore all the walkers hold exactly the same hills at any time, and
    only walker 0 writes the log (name) and the hills file (str_save).
    The walkers also cross-check the number of hills in every round; a
    mismatch (e.g. different restart files) raises RuntimeError in all of them.

    The hill rate is N / nstp (N = number of walkers), so with the same nstp
    the bias grows N times faster than in run; scale nstp (or wght, for
    flooding) accordingly if the same rate is wanted.

    Rules (otherwise the walkers dead-lock):
    * all the walkers must do the same number of gradient calls with
      deposit = True (same step_number, same deposit flag at any time)
    * all the walkers must be created with the same parameters
    * create the parallel object BEFORE the engines (fork after OpenMM/threads
      is unsafe) and re-seed numpy.random in every walker (the forked processes
      inherit the same random state)
    * call close() before opar.stop() (stop() ends the process at once)
    * write the final outputs (save_hills, get_pmf) from one walker only (node 0)

    Parameters (see run): colvar, sigm, wght, nstp, delta, temperature, wmin,
    wall, name, str_hills, str_save, nsav, plus
    opar    : qm3.utils.parallel.client_fsi / client_mpi object
    nsav counts hills (of all the walkers), not rounds.

    Log columns: step (gradient calls of each walker), walker, s_center, height,
    height_raw, V_bias (before the round).
    """
    def __init__( self, colvar: qm3.engines.mmres.colvar_s, opar: object,
            sigm: float, wght: float,
            nstp: typing.Optional[int] = 100,
            delta: typing.Optional[float] = None,
            temperature: typing.Optional[float] = None,
            wmin: typing.Optional[float] = None,
            wall: typing.Optional[list] = [],
            name: typing.Optional[str] = None,
            str_hills: typing.Optional[str] = "",
            str_save: typing.Optional[str] = "",
            nsav: typing.Optional[int] = 100 ):
        if( opar.ncpu < 1 or opar.node < 0 or opar.node >= opar.ncpu ):
            raise ValueError( "metadyn.run_mw: wrong parallel object (node %d of %d)"%( opar.node, opar.ncpu ) )
        self.opar  = opar
        self.nwalk = opar.ncpu
        self.walker = opar.node
        self.root  = ( opar.node == 0 )
        self.nin   = 0
        # hills are identical in every walker: only the root writes them
        run.__init__( self, colvar, sigm, wght, nstp, delta, temperature, wmin, wall,
                None, str_hills, str_save if self.root else "", nsav )
        if( name != None and self.root ):
            restart = ( str_hills != "" )
            self.fdsc = open( name, "at" if restart else "wt" )
            if( not restart ):
                self.fdsc.write( self._header() )
                self.fdsc.write( "#%20s%10s%20s%20s%20s%20s\n"%( "step", "walker", "s_center", "height", "height_raw", "V_bias" ) )
                self.fdsc.flush()


    def _header( self ) -> str:
        return( "# [%d walkers] "%( self.nwalk ) + run._header( self )[2:] )


    def _collect( self, cval: float ):
        self.step += 1
        if( len( self.wall ) == 3 ):
            depo = cval >= self.wall[1] and cval <= self.wall[2]
        else:
            depo = True
        if( depo ):
            self.nall += 1
            self.nin  += 1
            self.accu += cval
        if( self.step % self.nstp == 0 ):
            self._round()
            self.accu = 0.0
            self.nin  = 0


    def _round( self ):
        """
        synchronized deposition round (all the walkers must be here)
        """
        mine = [ 1.0 if self.nin > 0 else 0.0,
                 self.accu / self.nin if self.nin > 0 else 0.0,
                 float( self.nhill ) ]
        if( self.root ):
            info = [ mine ]
            for who in range( 1, self.nwalk ):
                info.append( self.opar.recv_r8( who, 3 ) )
            rep = [ 1.0 if all( int( round( x[2] ) ) == self.nhill for x in info ) else 0.0 ]
            for x in info:
                rep += [ x[0], x[1] ]
            for who in range( 1, self.nwalk ):
                self.opar.send_r8( who, rep )
        else:
            self.opar.send_r8( 0, mine )
            rep = self.opar.recv_r8( 0, 1 + 2 * self.nwalk )
        if( rep[0] < 0.5 ):
            raise RuntimeError( "metadyn.run_mw: the walkers hold a different number of hills (walker %d: %d)"%
                    ( self.walker, self.nhill ) )
        # heights with the bias before the round
        new = []
        for who in range( self.nwalk ):
            if( rep[1+2*who] > 0.5 ):
                sval = rep[2+2*who]
                vbar = float( self.get_bias( sval ) )
                wval, wraw = self._height( vbar )
                new.append( ( who, sval, wval, wraw, vbar ) )
        if( len( new ) == 0 ):
            return
        nbef = self.nhill
        for who, sval, wval, wraw, vbar in new:
            self.hill_s.append( sval )
            self.hill_w.append( wval )
            if( self.fdsc != None ):
                self.fdsc.write( "%20.6lf%10d%20.10lf%20.10lf%20.10lf%20.10lf\n"%
                                 ( float( self.step ), who, sval, wval, wraw, vbar ) )
        self._hs = numpy.array( self.hill_s, dtype=numpy.float64 )
        self._hw = numpy.array( self.hill_w, dtype=numpy.float64 )
        if( self.fdsc != None ):
            self.fdsc.flush()
        if( self.sfil != "" and self.nhill // self.nsav > nbef // self.nsav ):
            self.save_hills( self.sfil )
