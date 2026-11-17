import  math
import  os
import  numpy
import  typing
import  qm3.data


# =================================================================================================
# Targeted free energy perturbation (TFEP) for LL > HL corrections of free energy profiles
#
# Phys. Rev. E v65, p046122 (2002) [doi:10.1103/PhysRevE.65.046122]
# J. Phys. Chem. Lett. v12, p9449 (2021) [doi:10.1021/acs.jpclett.1c02135]
#
# Stage 1:  recorder      -> dataset ( x_i, s_i, d_i, E_LL, g_LL )
# Stage 2:  evaluate      -> E_HL, g_HL at x_i
#           gauss_map     -> s-preserving map M_s ( analytic, quasi-harmonic ) [or shift_map]
# Stage 3:  evaluate      -> E_HL at y_i = M( x_i )  ( independent evaluation set )
#           fes_correction / delta_f
#
# The map NEVER moves the atoms which define the collective variable ( nor the frame atoms ):
# d_c( M( x ) ) = d_c( x )  =>  s( M( x ) ) = s( x )  ( colvar_s depends ONLY on d ), which is
# the condition of eq. 16 of the paper. As a bonus, any restraint on d ( string windows,
# umbrellas ) is invariant, so it cancels in the work.
#
#   w( x ) = U_HL( M( x ) ) - U_LL( x ) - kT ln | J_M( x ) |
#   Df( s ) = - kT ln  ( 1 / N_bin ) SUM_i exp( - w_i / kT ) [s_i in bin]
#
# energies in kJ/mol, coordinates in Ang
# =================================================================================================


def kt_of( temperature: float ) -> float:
    return( 1.0e-3 * qm3.data.R * temperature )


def logsumexp( x: numpy.array ) -> float:
    x = numpy.ravel( x )
    if( x.size == 0 ):
        return( - numpy.inf )
    m = numpy.max( x )
    if( not numpy.isfinite( m ) ):
        return( m )
    return( float( m + math.log( numpy.sum( numpy.exp( x - m ) ) ) ) )


# =================================================================================================
# dataset
# =================================================================================================

class dataset( object ):
    """
    idx   (nat,)          global indices of the stored atoms (sorted)
    cvi   (ncv,)          global indices of the atoms defining the CV (fixed by the map)
    crd   (nf,nat,3)      coordinates
    s     (nf,)           colvar_s value (nan if there is no colvar yet)
    d     (nf,nc)         distances of the colvar
    win   (nf,)           window / string node index
    step  (nf,)           MD step index
    e_ll, g_ll            LL energy and gradient WITHOUT any bias (kJ/mol, kJ/mol/Ang)
    v_bias                bias energy subtracted from LL (kJ/mol)
    e_hl, g_hl            HL energy and gradient at the same frames (stage 2)
    """
    _arr = ( "idx", "cvi", "crd", "s", "d", "win", "step", "e_ll", "g_ll", "v_bias", "e_hl", "g_hl" )
    _per = ( "crd", "s", "d", "win", "step", "e_ll", "g_ll", "v_bias", "e_hl", "g_hl" )

    def __init__( self, **kwd ):
        for k in dataset._arr:
            setattr( self, k, kwd.get( k, None ) )

    def __len__( self ) -> int:
        return( 0 if self.crd is None else self.crd.shape[0] )

    def save( self, name: str ):
        numpy.savez( name, **{ k: getattr( self, k ) for k in dataset._arr if getattr( self, k ) is not None } )

    @staticmethod
    def load( name: str ) -> object:
        with numpy.load( name ) as f:
            return( dataset( **{ k: f[k] for k in f.files } ) )

    def subset( self, mask: numpy.array ) -> object:
        mask = numpy.asarray( mask )
        out = dataset( idx = self.idx, cvi = self.cvi )
        for k in dataset._per:
            v = getattr( self, k )
            if( v is not None ):
                setattr( out, k, v[mask] )
        return( out )

    @staticmethod
    def concat( lst: list ) -> object:
        out = dataset( idx = lst[0].idx, cvi = lst[0].cvi )
        for k in dataset._per:
            if( all( getattr( i, k ) is not None for i in lst ) ):
                setattr( out, k, numpy.concatenate( [ getattr( i, k ) for i in lst ], axis = 0 ) )
        return( out )

    def loc( self, atoms: numpy.array ) -> numpy.array:
        atoms = numpy.asarray( atoms, dtype=numpy.int64 )
        pos = numpy.searchsorted( self.idx, atoms )
        pos = numpy.minimum( pos, self.idx.shape[0] - 1 )
        if( not numpy.all( self.idx[pos] == atoms ) ):
            raise ValueError( "tfep.dataset: some atoms are not stored in the dataset" )
        return( pos )


## =================================================================================================
## stage 1: recorder
## =================================================================================================
#
#class recorder( object ):
#    """
#    Callback for qm3.actions.dynamics ( current_step = rec ) or for any loop calling rec( mol, step )
#    right after mol.get_grad(): it stores every 'stride' steps the frame and the LL energy and
#    gradient WITHOUT the bias.
#
#    mol.engines must hold the LL engines ( + MM ) and the bias engines ( string, cvs_umb, ... )
#
#        bias:    list of the bias objects ( the ones added to mol.engines ). They are re-evaluated
#                 alone ( cheap: they only depend on the CV ) and subtracted from mol.func / mol.grad.
#                 The accumulators of string are protected during the re-evaluation.
#        colvar:  qm3.engines.mmres.colvar_s ( s and d ) or, if None, pairs = [ [i,j], ... ] ( only d )
#        idx:     atoms to be stored ( default all ): at least the QM atoms + the MM atoms that the HL
#                 engine needs ( the rest is taken from the molecule when evaluating )
#        window:  window / string node index of this run ( it can be changed between runs )
#    """
#    def __init__( self, mol: object,
#            colvar: typing.Optional[object] = None,
#            pairs: typing.Optional[list] = None,
#            bias: typing.Optional[list] = [],
#            idx: typing.Optional[numpy.array] = None,
#            stride: typing.Optional[int] = 100,
#            skip: typing.Optional[int] = 0,
#            window: typing.Optional[int] = 0,
#            name: typing.Optional[str] = "",
#            nsav: typing.Optional[int] = 1000 ):
#        if( colvar is None and pairs is None ):
#            raise ValueError( "tfep.recorder: colvar or pairs are needed" )
#        self.cv = colvar
#        if( colvar is not None ):
#            self.pairs = [ [ int( i ), int( j ) ] for i, j in colvar.atom ]
#            self.cvi = numpy.array( sorted( colvar.indx ), dtype=numpy.int64 )
#        else:
#            self.pairs = [ [ int( i ), int( j ) ] for i, j in pairs ]
#            self.cvi = numpy.unique( numpy.array( self.pairs, dtype=numpy.int64 ) )
#        self.bias = list( bias )
#        self.idx = numpy.arange( mol.natm ) if idx is None else numpy.unique( numpy.asarray( idx, dtype=numpy.int64 ) )
#        if( not numpy.all( numpy.isin( self.cvi, self.idx ) ) ):
#            raise ValueError( "tfep.recorder: the CV atoms should be in idx" )
#        self.stride = int( stride )
#        self.skip = int( skip )
#        self.window = int( window )
#        self.name = name
#        self.nsav = max( 1, int( nsav ) )
#        self.buf = { k: [] for k in ( "crd", "s", "d", "win", "step", "e_ll", "g_ll", "v_bias" ) }
#
#
#    def __call__( self, mol: object, step: int ):
#        if( step <= self.skip or step % self.stride != 0 ):
#            return
#        if( self.cv is not None ):
#            sval, ccrd = self.cv.value( mol )
#        else:
#            sval = numpy.nan
#            ccrd = numpy.array( [ numpy.linalg.norm( mol.coor[j] - mol.coor[i] ) for i, j in self.pairs ] )
#        # bias alone: func / grad of the biases evaluated on a scratch
#        fref = mol.func
#        gref = mol.grad
#        vb = 0.0
#        gb = numpy.zeros_like( gref )
#        for b in self.bias:
#            acc = None
#            if( hasattr( b, "amet" ) and hasattr( b, "adif" ) and hasattr( b, "nstp" ) ):
#                acc = ( b.nstp, b.amet.copy(), b.adif.copy() )
#            mol.func = 0.0
#            mol.grad = numpy.zeros_like( gref )
#            b.get_grad( mol )
#            vb += mol.func
#            gb += mol.grad
#            if( acc is not None ):
#                b.nstp, b.amet, b.adif = acc
#        mol.func = fref
#        mol.grad = gref
#        gb *= mol.actv.astype( numpy.float64 )
#        self.buf["crd"].append( mol.coor[self.idx].copy() )
#        self.buf["s"].append( sval )
#        self.buf["d"].append( ccrd.copy() )
#        self.buf["win"].append( self.window )
#        self.buf["step"].append( step )
#        self.buf["e_ll"].append( fref - vb )
#        self.buf["g_ll"].append( ( gref - gb )[self.idx].copy() )
#        self.buf["v_bias"].append( vb )
#        if( self.name != "" and len( self.buf["s"] ) % self.nsav == 0 ):
#            self.save()
#
#
#    def get( self ) -> object:
#        return( dataset( idx = self.idx, cvi = self.cvi, **{ k: numpy.array( v ) for k, v in self.buf.items() } ) )
#
#
#    def save( self, name: typing.Optional[str] = "" ):
#        self.get().save( self.name if name == "" else name )
#
#
## =================================================================================================
## stage 2/3: engines on frames
## =================================================================================================
#
#def evaluate( mol: object, idx: numpy.array, crd: numpy.array,
#        grad: typing.Optional[bool] = False,
#        ckpt: typing.Optional[str] = "",
#        nsav: typing.Optional[int] = 10,
#        verbose: typing.Optional[bool] = True ) -> tuple:
#    """
#    Energy ( and gradient ) of mol.engines on the frames crd ( nf, len( idx ), 3 ).
#    mol.engines must be the HL set: SAME MM + QM-HL engine ( it should differ from the LL set only
#    in the QM engine, so that the MM part cancels in HL - LL ); no bias engines.
#    All the atoms are made active while evaluating.
#    ckpt: npz file, restartable.
#    returns ( e ( nf, ), g ( nf, len( idx ), 3 ) or None )
#    """
#    nf = crd.shape[0]
#    ene = numpy.full( nf, numpy.nan )
#    gra = numpy.full( ( nf, len( idx ), 3 ), numpy.nan ) if grad else None
#    if( ckpt != "" and os.path.isfile( ckpt ) ):
#        with numpy.load( ckpt ) as f:
#            if( f["e"].shape[0] == nf ):
#                ene = f["e"].copy()
#                if( grad and "g" in f.files ):
#                    gra = f["g"].copy()
#    c0 = mol.coor.copy()
#    a0 = mol.actv.copy()
#    mol.actv[:] = True
#    ndo = 0
#    try:
#        for i in range( nf ):
#            if( not numpy.isnan( ene[i] ) ):
#                continue
#            mol.coor[idx] = crd[i]
#            if( grad ):
#                ene[i] = mol.get_grad()
#                gra[i] = mol.grad[idx]
#            else:
#                ene[i] = mol.get_func()
#            ndo += 1
#            if( verbose ):
#                print( "[tfep.evaluate] %6d / %6d  %20.6lf"%( i + 1, nf, ene[i] ) )
#            if( ckpt != "" and ndo % nsav == 0 ):
#                numpy.savez( ckpt, e = ene, **( { "g": gra } if grad else {} ) )
#    finally:
#        mol.coor[:] = c0
#        mol.actv[:] = a0
#    if( ckpt != "" ):
#        numpy.savez( ckpt, e = ene, **( { "g": gra } if grad else {} ) )
#    return( ( ene, gra ) )


# =================================================================================================
# stage 2: analytic quasi-harmonic ( Gaussian > Gaussian ) s-preserving map
# =================================================================================================

def _sym_pow( mat: numpy.array, p: float ) -> numpy.array:
    e, u = numpy.linalg.eigh( 0.5 * ( mat + mat.T ) )
    return( numpy.dot( u * numpy.power( numpy.maximum( e, 1.e-300 ), p ), u.T ) )


class gauss_map( object ):
    """
    Linear map in the local frame of the mapped atoms ( q = R ( x - o ), frame defined by three FIXED atoms )
    conditioned on the features of the fixed atoms ( powers of s and the distances d ):

        p_A( q | phi ) ~ N( mu_A( phi ), S_A )                      ( from the LL samples )
        grad_q( U_HL - U_LL ) ~ g0 + G^T phi + H ( q - mu_A )      ( regression on the gradients: force matching of the difference )
        => p_B( q | phi ) ~ N( mu_B( phi ), S_B )      S_B^-1 = S_A^-1 + H/kT     mu_B = mu_A - S_B g / kT
        M:  q' = mu_B( phi ) + T ( q - mu_A( phi ) )     T = optimal-transport map between the two Gaussians
        ln J = 1/2 ( ln det S_B - ln det S_A )

    The fixed atoms ( CV atoms + frame atoms ) are never moved: d, s and any restraint on them are preserved
    EXACTLY, and J has no contribution from the conditioning ( block-triangular map ).
    The fit only needs ( x_i, g_LL, g_HL ) of the LL trajectory ( biased or not ): within a CV slice
    p( q | s ) does not depend on the bias.

        free:   global indices of the atoms to map ( e.g. QM atoms ); fixed atoms are removed from it
        frame:  3 global indices ( non collinear ), fixed by the map
        degree: degree of the polynomial in s
        shrink: shrinkage of the covariance towards its diagonal
        ridge:  ridge ( standardized units ) of the regression of the gradients
        emax:   eigenvalues of S_A^1/2 H S_A^1/2 are clipped to > - emax ( S_B positive definite )
        temperature: K ( g and H are divided by kT: the exponent is U / kT )
        damp:   0 < damp <= 1; the map is M_d = identity + d ( M - identity ): still an invertible affine map
                ( the Jacobian is recomputed ), just less ambitious; use it if the full map moves atoms too much
        use_fixed: also condition on the local coordinates of the fixed atoms ( CV + frame ): the energy difference
                depends on the geometry RELATIVE to them ( free-fixed couplings ), and the Jacobian does not change
        use_d:  False: condition only on s ( the distances d are strongly collinear with s )
        fridge: ridge ( x N ) of the regression of mu_A on the features ( raise it if the features are collinear )
        vmax:   Ang; only the directions of the LL covariance with std < vmax are mapped ( the soft ones, where
                the quasi-harmonic response is not reliable, are left untouched ): M = identity + U ( M_s - identity ) U^T
    """
    def __init__( self, free: numpy.array, frame: list,
            degree: typing.Optional[int] = 2,
            shrink: typing.Optional[float] = 0.05,
            ridge: typing.Optional[float] = 1.e-2,
            emax: typing.Optional[float] = 0.9,
            temperature: typing.Optional[float] = 300.0,
            damp: typing.Optional[float] = 1.0,
            vmax: typing.Optional[float] = None,
            use_d: typing.Optional[bool] = True,
            fridge: typing.Optional[float] = 1.e-6,
            use_fixed: typing.Optional[bool] = True ):
        self.use_fixed = use_fixed
        self.use_d = use_d
        self.fridge = fridge
        self.kt = kt_of( temperature )
        self.damp = float( damp )
        self.vmax = vmax
        self.free0 = numpy.unique( numpy.asarray( free, dtype=numpy.int64 ) )
        self.frame = [ int( i ) for i in frame ]
        if( len( self.frame ) != 3 ):
            raise ValueError( "tfep.gauss_map: frame needs three atoms" )
        self.degree = int( degree )
        self.shrink = shrink
        self.ridge = ridge
        self.emax = emax
        self.ready = False


    def _fixfeat( self, dat: object ) -> numpy.array:
        """ local coordinates of the fixed atoms, without the trivial ones ( origin, axis, plane ) """
        if( not self.use_fixed ):
            return( numpy.zeros( ( len( dat ), 0 ) ) )
        c = dat.crd
        a0, a1, a2 = dat.loc( self.frame )
        o  = c[:,a0,:]
        e1 = c[:,a1,:] - o
        e1 /= numpy.linalg.norm( e1, axis = 1 )[:,None]
        v  = c[:,a2,:] - o
        e2 = v - numpy.sum( v * e1, axis = 1 )[:,None] * e1
        e2 /= numpy.linalg.norm( e2, axis = 1 )[:,None]
        rot = numpy.stack( ( e1, e2, numpy.cross( e1, e2 ) ), axis = 1 )
        cols = []
        for a in self.fixed:
            if( a == self.frame[0] ):
                continue
            q = numpy.einsum( "nij,nj->ni", rot, c[:,dat.loc( [a] )[0],:] - o )
            if( a == self.frame[1] ):
                cols.append( q[:,0] )
            elif( a == self.frame[2] ):
                cols += [ q[:,0], q[:,1] ]
            else:
                cols += [ q[:,0], q[:,1], q[:,2] ]
        return( numpy.stack( cols, axis = 1 ) if cols else numpy.zeros( ( len( dat ), 0 ) ) )


    def _local( self, dat: object ) -> tuple:
        c = dat.crd
        a0, a1, a2 = dat.loc( self.frame )
        o  = c[:,a0,:]
        e1 = c[:,a1,:] - o
        e1 /= numpy.linalg.norm( e1, axis = 1 )[:,None]
        v  = c[:,a2,:] - o
        e2 = v - numpy.sum( v * e1, axis = 1 )[:,None] * e1
        nn = numpy.linalg.norm( e2, axis = 1 )
        if( numpy.min( nn ) < 1.e-3 ):
            raise ValueError( "tfep.gauss_map: frame atoms are (nearly) collinear" )
        e2 /= nn[:,None]
        e3 = numpy.cross( e1, e2 )
        rot = numpy.stack( ( e1, e2, e3 ), axis = 1 )                              # ( nf, 3, 3 ) rows = axes
        dx  = c[:,self.fl,:] - o[:,None,:]
        qq  = numpy.einsum( "nij,nfj->nfi", rot, dx )
        return( ( qq.reshape( c.shape[0], -1 ), rot, o ) )


    def _feat( self, dat: object ) -> numpy.array:
        cols = []
        if( self.use_s ):
            z = ( dat.s - self.s0 ) / self.ss
            for p in range( 1, self.degree + 1 ):
                cols.append( z ** p )
        for j in range( self.d0.shape[0] ):
            cols.append( ( dat.d[:,j] - self.d0[j] ) / self.sd[j] )
        ff = ( self._fixfeat( dat ) - self.f0 ) / self.fs
        if( len( cols ) == 0 and ff.shape[1] == 0 ):
            return( numpy.zeros( ( len( dat ), 0 ) ) )
        if( len( cols ) == 0 ):
            return( ff )
        return( numpy.hstack( ( numpy.stack( cols, axis = 1 ), ff ) ) )


    def fit( self, dat: object ):
        fixed = numpy.union1d( numpy.asarray( dat.cvi, dtype=numpy.int64 ), numpy.array( self.frame, dtype=numpy.int64 ) )
        self.free = numpy.setdiff1d( self.free0, fixed )
        self.fixed = [ int( i ) for i in fixed ]
        self.fl = dat.loc( self.free )
        self.f0 = numpy.zeros( 0 )
        self.fs = numpy.ones( 0 )
        _ff = self._fixfeat( dat )
        self.f0 = _ff.mean( axis = 0 )
        self.fs = numpy.maximum( _ff.std( axis = 0 ), 1.e-8 )
        if( dat.e_hl is None or dat.g_hl is None ):
            raise ValueError( "tfep.gauss_map.fit: the dataset needs g_hl ( run evaluate( ..., grad = True ) )" )
        if( numpy.max( numpy.abs( dat.g_ll[:,self.fl,:] ) ) == 0.0 ):
            raise ValueError( "tfep.gauss_map.fit: null LL gradient on the free atoms ( frozen atoms in the sampling? )" )
        nf = len( dat )
        self.use_s = not numpy.any( numpy.isnan( dat.s ) )
        self.s0 = float( numpy.mean( dat.s ) ) if self.use_s else 0.0
        self.ss = float( numpy.std( dat.s ) ) if self.use_s else 1.0
        self.d0 = numpy.mean( dat.d, axis = 0 ) if self.use_d else numpy.zeros( 0 )
        self.sd = numpy.maximum( numpy.std( dat.d, axis = 0 ), 1.e-8 ) if self.use_d else numpy.ones( 0 )
        phi = self._feat( dat )
        qq, rot, o = self._local( dat )
        mm = qq.shape[1]
        gd = ( dat.g_hl - dat.g_ll )[:,self.fl,:]
        gq = numpy.einsum( "nij,nfj->nfi", rot, gd ).reshape( nf, -1 )
        # mu_A( phi )
        xx = numpy.hstack( ( numpy.ones( ( nf, 1 ) ), phi ) )
        pen = numpy.eye( xx.shape[1] ) * self.fridge * nf
        pen[0,0] = 0.0
        self.bA = numpy.linalg.solve( numpy.dot( xx.T, xx ) + pen, numpy.dot( xx.T, qq ) )
        rr = qq - numpy.dot( xx, self.bA )
        cov = numpy.dot( rr.T, rr ) / max( 1.0, nf - xx.shape[1] )
        sA = ( 1.0 - self.shrink ) * cov + self.shrink * numpy.diag( numpy.diag( cov ) )
        sig = numpy.sqrt( numpy.maximum( numpy.diag( cov ), 1.e-16 ) )
        # grad( HL - LL ) = g0 + G^T phi + H r
        zz = numpy.hstack( ( xx, rr / sig ) )
        pen = numpy.eye( zz.shape[1] ) * self.ridge * nf
        pen[0,0] = 0.0
        gam = numpy.linalg.solve( numpy.dot( zz.T, zz ) + pen, numpy.dot( zz.T, gq ) )
        self.gg = gam[:xx.shape[1]] / self.kt                                      # ( 1 + nfeat, m ): g0 and G  [1/Ang]
        hh = ( gam[xx.shape[1]:] / sig[:,None] ).T / self.kt                       # H_ij = gam[j,i] / sig_j  [1/Ang^2]
        hh = 0.5 * ( hh + hh.T )
        # stiff subspace
        if( self.vmax is None ):
            uu = numpy.eye( mm )
        else:
            ev, uv = numpy.linalg.eigh( sA )
            uu = uv[:,ev < self.vmax ** 2]
            if( uu.shape[1] == 0 ):
                raise ValueError( "tfep.gauss_map: no direction with std < vmax" )
        self.uu = uu
        ms = uu.shape[1]
        sA_ = numpy.dot( uu.T, numpy.dot( sA, uu ) )
        hh_ = numpy.dot( uu.T, numpy.dot( hh, uu ) )
        # Gaussian B ( in the subspace )
        sh = _sym_pow( sA_, 0.5 )
        mat = numpy.dot( sh, numpy.dot( hh_, sh ) )
        e, u = numpy.linalg.eigh( 0.5 * ( mat + mat.T ) )
        e = numpy.maximum( e, - self.emax )
        sB = numpy.dot( sh, numpy.dot( u / ( 1.0 + e ), numpy.dot( u.T, sh ) ) )
        sB = 0.5 * ( sB + sB.T )
        ish = _sym_pow( sA_, - 0.5 )
        mid = _sym_pow( numpy.dot( sh, numpy.dot( sB, sh ) ), 0.5 )
        self.tt = numpy.dot( ish, numpy.dot( mid, ish ) )                          # optimal-transport map
        self.sB = sB
        self.sA = sA
        print( "[gauss_map] mapped directions: %d of %d"%( ms, mm ) )
        self.tt_eff = ( 1.0 - self.damp ) * numpy.eye( ms ) + self.damp * self.tt
        sgn, ldt = numpy.linalg.slogdet( self.tt_eff )
        if( sgn <= 0 ):
            raise ValueError( "tfep.gauss_map: singular map, decrease damp" )
        self.logj = float( ldt )
        self.H = hh
        self.ready = True
        # diagnostics: how much the map moves things and how much energy it is expected to gain
        gm = numpy.dot( xx, self.gg )
        dq = ( numpy.dot( numpy.dot( rr, uu ), ( self.tt_eff - numpy.eye( ms ) ).T ) - self.damp * numpy.dot( numpy.dot( gm, uu ), sB.T ) ) @ uu.T
        self.rms_shift = float( numpy.sqrt( numpy.mean( numpy.sum( dq.reshape( nf, -1, 3 ) ** 2, axis = 2 ) ) ) )
        print( "[gauss_map] free atoms: %d  frames: %d  rms displacement: %.4lf Ang  ln J: %.4lf"%(
            len( self.free ), nf, self.rms_shift, self.logj ) )


    def parts( self, dat: object ) -> dict:
        """ diagnostics: size of the features, of the residual r = q - mu_A( phi ), and of the two pieces of the displacement """
        nf = len( dat )
        phi = self._feat( dat )
        xx = numpy.hstack( ( numpy.ones( ( nf, 1 ) ), phi ) )
        qq, rot, o = self._local( dat )
        muA = numpy.dot( xx, self.bA )
        r = qq - muA
        rs = numpy.dot( r, self.uu )
        sh_ = - self.damp * numpy.dot( numpy.dot( numpy.dot( xx, self.gg ), self.uu ), self.sB.T )
        lin = numpy.dot( rs, ( self.tt_eff - numpy.eye( rs.shape[1] ) ).T )
        return( { "phi_max": float( numpy.abs( phi ).max() ) if phi.shape[1] else 0.0,
                  "r_rms": float( numpy.sqrt( numpy.mean( r ** 2 ) ) ), "r_max": float( numpy.abs( r ).max() ),
                  "shift_rms": float( numpy.sqrt( numpy.mean( sh_ ** 2 ) ) ), "shift_max": float( numpy.abs( sh_ ).max() ),
                  "lin_rms": float( numpy.sqrt( numpy.mean( lin ** 2 ) ) ), "lin_max": float( numpy.abs( lin ).max() ) } )


    def forward( self, dat: object ) -> tuple:
        """ returns ( crd_mapped ( nf, nat, 3 ), logJ ( nf, ) ) """
        if( not self.ready ):
            raise Exception( "tfep.gauss_map: fit() first" )
        nf = len( dat )
        phi = self._feat( dat )
        xx = numpy.hstack( ( numpy.ones( ( nf, 1 ) ), phi ) )
        qq, rot, o = self._local( dat )
        muA = numpy.dot( xx, self.bA )
        rs = numpy.dot( qq - muA, self.uu )
        sh_ = - numpy.dot( numpy.dot( numpy.dot( xx, self.gg ), self.uu ), self.sB.T )
        qn = qq + ( self.damp * sh_ + numpy.dot( rs, ( self.tt_eff - numpy.eye( rs.shape[1] ) ).T ) ) @ self.uu.T
        qn = qn.reshape( nf, -1, 3 )
        xn = o[:,None,:] + numpy.einsum( "nji,nfj->nfi", rot, qn )
        out = dat.crd.copy()
        out[:,self.fl,:] = xn
        return( ( out, numpy.full( nf, self.logj ) ) )


# =================================================================================================
# stage 3: estimators
# =================================================================================================

def profile( s: numpy.array, edges: numpy.array, temperature: float ) -> tuple:
    """ F_A( s ) = - kT ln N_bin shifted to its minimum """
    kt = kt_of( temperature )
    cen = 0.5 * ( edges[1:] + edges[:-1] )
    hh = numpy.histogram( s, bins = edges )[0].astype( numpy.float64 )
    ff = - kt * numpy.log( numpy.where( hh > 0.0, hh, numpy.nan ) )
    return( cen, ff - numpy.nanmin( ff ) )


def work( dat: object, e_hl_map: numpy.array, logj: numpy.array, temperature: float ) -> numpy.array:
    """ w = U_HL( M( x ) ) - U_LL( x ) - kT ln J    [kJ/mol] """
    return( e_hl_map - dat.e_ll - kt_of( temperature ) * logj )


def work_fep( dat: object, e_hl: numpy.array ) -> numpy.array:
    """ plain FEP: w = U_HL( x ) - U_LL( x ) """
    return( e_hl - dat.e_ll )


def delta_f( w: numpy.array, temperature: float,
        mask: typing.Optional[numpy.array] = None ) -> tuple:
    """
    Df = - kT ln ( 1 / N ) SUM exp( - w_i / kT )     ( eq. 6 / 17 of the paper, bin = mask )
    returns ( Df, ESS ) with ESS = ( SUM t_i )^2 / SUM t_i^2, t_i = exp( - w_i / kT ): effective number of
    samples of the exponential average ( THE convergence diagnostic: the closer to N the better )
    """
    kt = kt_of( temperature )
    if( mask is not None ):
        w = w[mask]
    n = w.shape[0]
    if( n == 0 ):
        return( ( numpy.nan, 0.0 ) )
    lt = - w / kt
    ess = math.exp( 2.0 * logsumexp( lt ) - logsumexp( 2.0 * lt ) )
    return( ( - kt * ( logsumexp( lt ) - math.log( n ) ), ess ) )


def fes_correction( s: numpy.array, w: numpy.array, edges: numpy.array,
        temperature: float,
        nboot: typing.Optional[int] = 0,
        block: typing.Optional[int] = 1,
        seed: typing.Optional[int] = 0 ) -> dict:
    """
    Df( s ) on the bins defined by edges ( eq. 17 ). Error: block bootstrap ( block = consecutive frames,
    >= correlation time in frames ). Also valid for 'states' ( edges = [ lo, hi ] ), and for the windows
    of a string ( s = dat.win, edges = np.arange( nw + 1 ) - 0.5 ).
    """
    nb = len( edges ) - 1
    cen = 0.5 * ( edges[1:] + edges[:-1] )
    ind = numpy.clip( numpy.digitize( s, edges ) - 1, -1, nb )
    df = numpy.full( nb, numpy.nan )
    es = numpy.zeros( nb )
    nn = numpy.zeros( nb, dtype=numpy.int64 )
    for b in range( nb ):
        m = ind == b
        nn[b] = m.sum()
        df[b], es[b] = delta_f( w, temperature, m )
    err = numpy.full( nb, numpy.nan )
    if( nboot > 1 ):
        rng = numpy.random.default_rng( seed )
        n = s.shape[0]
        nblk = int( math.ceil( n / float( block ) ) )
        bt = numpy.full( ( nboot, nb ), numpy.nan )
        for k in range( nboot ):
            ids = ( rng.integers( 0, nblk, nblk )[:,None] * block + numpy.arange( block )[None,:] ).ravel()
            ids = ids[ids < n]
            for b in range( nb ):
                m = ind[ids] == b
                if( m.sum() > 0 ):
                    bt[k,b] = delta_f( w[ids], temperature, m )[0]
        err = numpy.nanstd( bt, axis = 0 )
    return( { "s": cen, "n": nn, "ess": es, "df": df, "err": err } )


# =================================================================================================
# stage 2 ( alternative ): shift map fitted by minimizing the variance of w
# =================================================================================================

class shift_map( gauss_map ):
    """
    y_q = q + C phi1( fixed atoms )      ( translation of the free atoms in the local frame, conditioned on the
                                          features of the fixed atoms only  =>  ln J = 0 EXACTLY, s and d preserved )

    To first order the work is  w = dU( x ) + g_HL( x ) . delta + 1/2 delta^T K delta,  with dU = U_HL - U_LL,
    delta = C phi1 and K = kT S_A^-1 the ( quasi-harmonic ) cost of moving the atoms. C is the minimizer of
    the within-window variance of w: a linear regression ( g_HL (x) phi1 as features ) with a penalty on the
    energetic cost of the displacement ( mu ); mu is chosen by cross validation ( first / second half of every
    window ) on the predicted w, so no HL energy on mapped geometries is needed to fit it.
    It does not assume that the LL distribution is gaussian ( only the cost term uses S_A ).

        mus:   candidate penalties ( in units of kT cost per kJ/mol of squared work )
    """
    def __init__( self, free: numpy.array, frame: list,
            degree: typing.Optional[int] = 2,
            shrink: typing.Optional[float] = 0.05,
            temperature: typing.Optional[float] = 300.0,
            use_d: typing.Optional[bool] = True,
            use_fixed: typing.Optional[bool] = True,
            fridge: typing.Optional[float] = 1.e-6,
            mus: typing.Optional[tuple] = ( 1.e-3, 3.e-3, 1.e-2, 3.e-2, 1.e-1, 3.e-1, 1.0, 3.0 ) ):
        super().__init__( free, frame, degree = degree, shrink = shrink, temperature = temperature,
                use_d = use_d, use_fixed = use_fixed, fridge = fridge )
        self.mus = mus
        self.logj = 0.0


    def _prep( self, dat: object ):
        fixed = numpy.union1d( numpy.asarray( dat.cvi, dtype=numpy.int64 ), numpy.array( self.frame, dtype=numpy.int64 ) )
        self.free = numpy.setdiff1d( self.free0, fixed )
        self.fixed = [ int( i ) for i in fixed ]
        self.fl = dat.loc( self.free )
        self.f0 = numpy.zeros( 0 )
        self.fs = numpy.ones( 0 )
        _ff = self._fixfeat( dat )
        self.f0 = _ff.mean( axis = 0 )
        self.fs = numpy.maximum( _ff.std( axis = 0 ), 1.e-8 )
        self.use_s = not numpy.any( numpy.isnan( dat.s ) )
        self.s0 = float( numpy.mean( dat.s ) ) if self.use_s else 0.0
        self.ss = float( numpy.std( dat.s ) ) if self.use_s else 1.0
        self.d0 = numpy.mean( dat.d, axis = 0 ) if self.use_d else numpy.zeros( 0 )
        self.sd = numpy.maximum( numpy.std( dat.d, axis = 0 ), 1.e-8 ) if self.use_d else numpy.ones( 0 )


    @staticmethod
    def _wcen( a: numpy.array, win: numpy.array ) -> numpy.array:
        a = a.copy()
        for w in numpy.unique( win ):
            m = win == w
            a[m] -= a[m].mean( axis = 0 )
        return( a )


    def _design( self, dat: object ) -> tuple:
        qq, rot, o = self._local( dat )
        gb = numpy.einsum( "nij,nfj->nfi", rot, dat.g_hl[:,self.fl,:] ).reshape( len( dat ), -1 )
        phi1 = numpy.hstack( ( numpy.ones( ( len( dat ), 1 ) ), self._feat( dat ) ) )
        return( ( qq, gb, phi1 ) )


    def fit( self, dat: object ):
        if( dat.e_hl is None or dat.g_hl is None or dat.win is None or dat.step is None ):
            raise ValueError( "tfep.shift_map.fit: e_hl, g_hl, win and step are needed" )
        self._prep( dat )
        nf = len( dat )
        qq, gb, phi1 = self._design( dat )
        mm, nk = gb.shape[1], phi1.shape[1]
        # LL covariance of the residual of q ( cost of moving atoms )
        pen = numpy.eye( nk ) * self.fridge * nf
        pen[0,0] = 0.0
        self.bA = numpy.linalg.solve( numpy.dot( phi1.T, phi1 ) + pen, numpy.dot( phi1.T, qq ) )
        rr = qq - numpy.dot( phi1, self.bA )
        cov = numpy.dot( rr.T, rr ) / max( 1.0, nf - nk )
        self.sA = ( 1.0 - self.shrink ) * cov + self.shrink * numpy.diag( numpy.diag( cov ) )
        kk = self.kt * numpy.linalg.inv( self.sA )                                # kJ/mol/Ang^2
        pp = numpy.kron( kk, numpy.dot( phi1.T, phi1 ) / nf )                      # mean( delta^T K delta ) = c^T P c
        du = dat.e_hl - dat.e_ll
        ff = ( gb[:,:,None] * phi1[:,None,:] ).reshape( nf, -1 )
        def solve( idx, mu ):
            yc = self._wcen( du[idx][:,None], dat.win[idx] )[:,0]
            fc = self._wcen( ff[idx], dat.win[idx] )
            a = numpy.dot( fc.T, fc ) / len( idx ) + mu * self.kt * pp + 1.e-9 * numpy.eye( pp.shape[0] ) * numpy.trace( pp ) / pp.shape[0]
            return( numpy.linalg.solve( a, - numpy.dot( fc.T, yc ) / len( idx ) ) )
        def predict( c, idx ):
            dl = numpy.dot( phi1[idx], c.reshape( mm, nk ).T )
            w = du[idx] + numpy.sum( gb[idx] * dl, axis = 1 ) + 0.5 * numpy.einsum( "ni,ij,nj->n", dl, kk, dl )
            return( self._wcen( w[:,None], dat.win[idx] )[:,0].std() )
        # cross validation: first / second half of every window ( by step )
        h1 = numpy.zeros( nf, dtype=numpy.bool_ )
        for w in numpy.unique( dat.win ):
            ii = numpy.flatnonzero( dat.win == w )
            ii = ii[numpy.argsort( dat.step[ii] )]
            h1[ii[:len( ii ) // 2]] = True
        i1, i2 = numpy.flatnonzero( h1 ), numpy.flatnonzero( ~h1 )
        res = []
        for mu in self.mus:
            e = 0.5 * ( predict( solve( i1, mu ), i2 ) + predict( solve( i2, mu ), i1 ) )
            res.append( e )
        self.cv = dict( zip( self.mus, res ) )
        self.mu = self.mus[int( numpy.argmin( res ) )]
        self.cvstd = float( numpy.min( res ) )
        self.C = solve( numpy.arange( nf ), self.mu ).reshape( mm, nk )
        self.fit_std = self._wcen( du[:,None], dat.win )[:,0].std()
        dl = numpy.dot( phi1, self.C.T )
        self.rms_shift = float( numpy.sqrt( numpy.mean( numpy.sum( dl.reshape( nf, -1, 3 ) ** 2, axis = 2 ) ) ) )
        self.kk = kk
        self.ready = True
        print( "[shift_map] free atoms: %d  frames: %d  mu: %g  cross-validated std( w ): %.2lf  ( without map: %.2lf ) kJ/mol  rms displacement: %.4lf Ang"%(
            len( self.free ), nf, self.mu, self.cvstd, self.fit_std, self.rms_shift ) )


    def predict_w( self, dat: object ) -> numpy.array:
        """ second order surrogate of the work ( no HL on mapped geometries ) """
        qq, gb, phi1 = self._design( dat )
        dl = numpy.dot( phi1, self.C.T )
        return( dat.e_hl - dat.e_ll + numpy.sum( gb * dl, axis = 1 ) + 0.5 * numpy.einsum( "ni,ij,nj->n", dl, self.kk, dl ) )


    def parts( self, dat: object ) -> dict:
        qq, gb, phi1 = self._design( dat )
        dl = numpy.dot( phi1, self.C.T )
        r = qq - numpy.dot( phi1, self.bA )
        return( { "phi_max": float( numpy.abs( phi1[:,1:] ).max() ) if phi1.shape[1] > 1 else 0.0,
                  "r_rms": float( numpy.sqrt( numpy.mean( r ** 2 ) ) ), "r_max": float( numpy.abs( r ).max() ),
                  "shift_rms": float( numpy.sqrt( numpy.mean( dl ** 2 ) ) ), "shift_max": float( numpy.abs( dl ).max() ),
                  "lin_rms": 0.0, "lin_max": 0.0 } )


    def forward( self, dat: object ) -> tuple:
        if( not self.ready ):
            raise Exception( "tfep.shift_map: fit() first" )
        nf = len( dat )
        qq, rot, o = self._local( dat )
        phi1 = numpy.hstack( ( numpy.ones( ( nf, 1 ) ), self._feat( dat ) ) )
        qn = ( qq + numpy.dot( phi1, self.C.T ) ).reshape( nf, -1, 3 )
        xn = o[:,None,:] + numpy.einsum( "nji,nfj->nfi", rot, qn )
        out = dat.crd.copy()
        out[:,self.fl,:] = xn
        return( ( out, numpy.zeros( nf ) ) )
