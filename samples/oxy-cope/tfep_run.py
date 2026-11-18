#!/usr/bin/env python3
"""
TFEP LL > HL for a string: fit the map, HL energies on the mapped geometries, corrections

  ./run.py fit      split train / evaluation, fit the map ( MAP=shift | gauss ), checks  -> output.tfep_map.npz, output.tfep_map.pk
  ./run.py hl       E_HL( M( x ) ) on the evaluation frames ( restartable )              -> output.hl_map.npz
  ./run.py fes      work, Df per window ( TFEP and plain FEP )                           -> output.tfep_fes.txt
  ./run.py explain  no engines: share of within-window dU = E_HL - E_LL explained by solute coordinates

input:   data.npz ( spe.py: needs idx, cvi, crd, s, d, win, step, e_ll, g_ll, e_hl, g_hl )
         sX/dcd.NNN  ( the trajectories that produced data.npz, in the same order )
"""
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  sys
import  itertools
import  pickle
import  numpy
import  qm3.utils.tfep


def _env( name: str, default ):
    v = os.environ.get( name )
    if( v is None ):
        return( default )
    if( v.lower() == "none" ):
        return( None )
    if( isinstance( default, bool ) ):
        return( v.lower() in ( "1", "true", "yes" ) )
    if( isinstance( default, int ) ):
        return( int( v ) )
    return( float( v ) )


NWIN        = 42          # windows ( string nodes )
TEMP        = 300.0
TRAIN_FRAC  = 0.5         # first part of every window -> training, the rest -> evaluation
DEGREE      = _env( "DEGREE", 2 )           # polynomial degree in s of the conditioning
SHRINK      = _env( "SHRINK", 0.05 )
RIDGE       = _env( "RIDGE", 1.e-2 )
USE_D       = _env( "USE_D", True )         # False: condition only on s ( if the distances d are collinear )
USE_FIXED   = _env( "USE_FIXED", True )     # condition also on the local coordinates of the fixed ( CV ) atoms
FRIDGE      = _env( "FRIDGE", 1.e-6 )       # ridge of the regression of the mean on the features ( raise it, e.g. 1e-2 )
VMAX        = _env( "VMAX", None )          # Ang: map only the directions with std < VMAX ( e.g. 0.15 ); None = all
DAMP        = _env( "DAMP", 1.0 )           # 0 < DAMP <= 1: M_d = identity + DAMP ( M - identity ), valid for any value
MAXDISP     = _env( "MAXDISP", 0.6 )        # Ang: stop if any atom is displaced more than this by the map
NBOOT       = 200
BLOCK       = 5           # frames per bootstrap block ( >= correlation time in frames )
EMAX        = _env( "EMAX", 0.9 )
MAPTYPE     = os.environ.get( "MAP", "shift" )   # "shift": translation fitted on variance of w ( J = 1 ); "gauss": quasi-harmonic Gaussian map
TAG         = os.environ.get( "TFEP_TAG", "" )
TAG         = ( "." + TAG ) if TAG != "" else ""
DAT         = os.environ.get( "DATA", "data.npz" )
MAPNPZ      = "output.tfep_map%s.npz"%TAG
MAPPK       = "output.tfep_map%s.pk"%TAG
HLNPZ       = "output.hl_map%s.npz"%TAG
OUT         = "output.tfep_fes%s.txt"%TAG


# -------------------------------------------------------------------------------------------------
# no molecule / engines needed
# -------------------------------------------------------------------------------------------------

def pick_frame( dat: object ) -> list:
    """ three CV atoms as non collinear as possible ( the map keeps them fixed ) """
    cand = [ int( i ) for i in dat.cvi ]
    if( len( cand ) < 3 ):
        raise ValueError( "less than three CV atoms: define the frame by hand" )
    best = None
    for tri in itertools.combinations( cand, 3 ):
        a, b, c = dat.loc( tri )
        v1 = dat.crd[:,b] - dat.crd[:,a]
        v2 = dat.crd[:,c] - dat.crd[:,a]
        sn = numpy.linalg.norm( numpy.cross( v1, v2 ), axis = 1 ) / ( numpy.linalg.norm( v1, axis = 1 ) * numpy.linalg.norm( v2, axis = 1 ) )
        if( best is None or sn.min() > best[0] ):
            best = ( sn.min(), list( tri ) )
    print( "frame atoms: %s  ( min sin(angle) = %.3lf )"%( best[1], best[0] ) )
    if( best[0] < 0.1 ):
        print( "WARNING: the frame atoms are nearly collinear in some frames" )
    return( best[1] )


def split( dat: object ) -> numpy.array:
    """ per window: first TRAIN_FRAC of the frames ( by step ) for training """
    if( dat.step is None ):
        raise ValueError( "data.npz has no 'step' ( frame number inside each dcd )" )
    tr = numpy.zeros( len( dat ), dtype=numpy.bool_ )
    for w in numpy.unique( dat.win ):
        ii = numpy.flatnonzero( dat.win == w )
        ii = ii[numpy.argsort( dat.step[ii] )]
        tr[ii[:int( numpy.ceil( len( ii ) * TRAIN_FRAC ) )]] = True
    return( tr )


def split_fit( dat: object ) -> tuple:
    ext = dat.crd.max( axis = 1 ) - dat.crd.min( axis = 1 )
    print( "frames: %d  windows: %d  max extent of the stored atoms: %.1lf Ang"%(
        len( dat ), len( numpy.unique( dat.win ) ), ext.max() ) )
    dg = numpy.abs( dat.g_hl - dat.g_ll ).max()
    print( "max |g_HL - g_LL| = %.3lf kJ/mol/Ang"%dg )
    if( dg == 0.0 ):
        raise ValueError( "g_hl == g_ll: the dataset was not built with HL data" )
    tr = split( dat )
    trn = dat.subset( tr )
    evl = dat.subset( ~tr )
    print( "training: %d frames  evaluation: %d frames"%( len( trn ), len( evl ) ) )
    frm = pick_frame( dat )
    if( MAPTYPE == "gauss" ):
        mp = qm3.utils.tfep.gauss_map( free = dat.idx, frame = frm, degree = DEGREE,
                shrink = SHRINK, ridge = RIDGE, temperature = TEMP, damp = DAMP, vmax = VMAX, use_d = USE_D, fridge = FRIDGE, emax = EMAX, use_fixed = USE_FIXED )
    elif( MAPTYPE == "shift" ):
        mp = qm3.utils.tfep.shift_map( free = dat.idx, frame = frm, degree = DEGREE, shrink = SHRINK, temperature = TEMP,
                use_d = USE_D, use_fixed = USE_FIXED, fridge = FRIDGE )
    else:
        raise ValueError( f"unknown MAP={MAPTYPE!r} (use 'shift' or 'gauss')" )
    mp.fit( trn )
    ym, lj = mp.forward( evl )
    if( hasattr( mp, "predict_w" ) ):
        sd0 = numpy.mean( [ ( evl.e_hl - evl.e_ll )[evl.win == w].std() for w in numpy.unique( evl.win ) ] )
        sd1 = numpy.mean( [ mp.predict_w( evl )[evl.win == w].std() for w in numpy.unique( evl.win ) ] )
        print( "evaluation set, within-window std( w ) ( kJ/mol ): plain FEP %.2lf  |  predicted with the map ( 2nd order surrogate ) %.2lf"%( sd0, sd1 ) )
    disp = numpy.linalg.norm( ym - evl.crd, axis = 2 )
    print( "displacement ( Ang ): rms %.3lf  max %.3lf"%( numpy.sqrt( numpy.mean( disp[:,mp.fl] ** 2 ) ), disp.max() ) )
    if( disp.max() > MAXDISP ):
        diagnose( trn, evl, mp, disp, frm )
        raise ValueError( "the map moves some atom more than MAXDISP ( see the diagnostics above )" )
    return( ( tr, mp, evl, ym, lj ) )


def diagnose( trn: object, evl: object, mp: object, disp: numpy.array, frm: list ):
    """ where does the displacement come from, and which settings tame it """
    print( "\n---- diagnostics ----" )
    print( "free atoms: %d ( %d coordinates ) for %d training frames ( ratio %.1lf )"%(
        len( mp.free ), 3 * len( mp.free ), len( trn ), len( trn ) / ( 3.0 * len( mp.free ) ) ) )
    mx = disp.max( axis = 0 )
    top = numpy.argsort( mx )[::-1][:6]
    print( "atoms with the largest displacement ( global index: max / rms Ang ):" )
    for k in top:
        print( "   %6d  %.3lf  %.3lf"%( evl.idx[k], mx[k], numpy.sqrt( numpy.mean( disp[:,k] ** 2 ) ) ) )
    ww = numpy.unique( evl.win )
    pw = numpy.array( [ disp[evl.win == w].max() for w in ww ] )
    print( "windows with the largest displacement ( window: max Ang ):", ", ".join( "%d: %.2lf"%( ww[i], pw[i] ) for i in numpy.argsort( pw )[::-1][:6] ) )
    for nom, dd in ( ( "training  ", trn ), ( "evaluation", evl ) ):
        p = mp.parts( dd )
        print( "%s: |phi|max %.1lf | residual r: rms %.3lf max %.3lf | shift: rms %.3lf max %.3lf | linear part: rms %.3lf max %.3lf  ( Ang )"%(
            nom, p["phi_max"], p["r_rms"], p["r_max"], p["shift_rms"], p["shift_max"], p["lin_rms"], p["lin_max"] ) )
    if( not hasattr( mp, "tt" ) ):
        print( "( shift map: mu = %g, see the cross-validated std of w above )"%mp.mu )
        print( "---- end ----\n" )
        return
    sd = numpy.sqrt( numpy.linalg.eigvalsh( mp.sA ) )
    print( "std of the LL covariance eigen-directions ( Ang ): min %.3lf  median %.3lf  max %.3lf  ( # > 0.15: %d of %d )"%(
        sd.min(), numpy.median( sd ), sd.max(), int( ( sd > 0.15 ).sum() ), sd.shape[0] ) )
    gd = ( trn.g_hl - trn.g_ll )[:,mp.fl,:]
    print( "g_HL - g_LL on the free atoms ( kJ/mol/Ang ): rms %.1lf  mean-force norm per atom: max %.1lf"%(
        numpy.sqrt( numpy.mean( numpy.sum( gd ** 2, axis = 2 ) ) ), numpy.linalg.norm( gd.mean( axis = 0 ), axis = 1 ).max() ) )
    print( "ln J = %.3lf"%mp.logj )
    print( "\n  vmax   damp |  rms    max   ( evaluation set; shrink / ridge as in the settings )" )
    for d in ( 1.0, 0.5 ):
        for vm in ( None, 0.3, 0.15, 0.08 ):
            try:
                m2 = qm3.utils.tfep.gauss_map( free = trn.idx, frame = frm, degree = DEGREE, shrink = SHRINK, ridge = RIDGE,
                        temperature = TEMP, damp = d, vmax = vm, emax = EMAX, use_d = USE_D, fridge = FRIDGE, use_fixed = USE_FIXED )
                m2.fit( trn )
                y2, _ = m2.forward( evl )
                d2 = numpy.linalg.norm( y2 - evl.crd, axis = 2 )
                print( " %5s %5.2lf | %6.3lf %6.3lf"%( vm, d, numpy.sqrt( numpy.mean( d2[:,m2.fl] ** 2 ) ), d2.max() ) )
            except Exception as e:
                print( " %5s %5.2lf | failed: %s"%( vm, d, e ) )
    print( "---- end ----\n" )


def report( evl: object, e_map: numpy.array, lj: numpy.array ) -> str:
    if( numpy.any( numpy.isnan( e_map ) ) ):
        raise ValueError( "E_HL on mapped frames is incomplete: run 'hl' again" )
    nwin = max( NWIN, int( evl.win.max() ) + 1 )
    w_t = qm3.utils.tfep.work( evl, e_map, lj, TEMP )
    w_f = qm3.utils.tfep.work_fep( evl, evl.e_hl )
    edges = numpy.arange( nwin + 1 ) - 0.5
    ws = evl.win.astype( numpy.float64 )
    ft = qm3.utils.tfep.fes_correction( ws, w_t, edges, TEMP, nboot = NBOOT, block = BLOCK )
    ff = qm3.utils.tfep.fes_correction( ws, w_f, edges, TEMP, nboot = NBOOT, block = BLOCK )
    ok = numpy.flatnonzero( ft["n"] > 0 )
    r0 = ok[0]
    s_mean = numpy.array( [ evl.s[evl.win == i].mean() if( ft["n"][i] > 0 ) else numpy.nan for i in range( nwin ) ] )
    sdt = numpy.array( [ w_t[evl.win == i].std() if( ft["n"][i] > 1 ) else numpy.nan for i in range( nwin ) ] )
    sdf = numpy.array( [ w_f[evl.win == i].std() if( ft["n"][i] > 1 ) else numpy.nan for i in range( nwin ) ] )
    out = "# df: correction of the restrained window ensembles ( kJ/mol ); rel: df - df[first window] ( = F_HL - F_LL profile shift )\n"
    out += "# win    n      <s> |  TFEP:  df   rel   err    ESS |  FEP:  df   rel   err    ESS | within-window std( w ): TFEP  FEP\n"
    for i in ok:
        out += "%5d %4d %8.4lf | %10.3lf %9.3lf %6.3lf %6.1lf | %10.3lf %9.3lf %6.3lf %6.1lf | %8.2lf %8.2lf\n"%(
            i, ft["n"][i], s_mean[i], ft["df"][i], ft["df"][i] - ft["df"][r0], ft["err"][i], ft["ess"][i],
            ff["df"][i], ff["df"][i] - ff["df"][r0], ff["err"][i], ff["ess"][i], sdt[i], sdf[i] )
    out += "# pooled std( w ): TFEP %.3lf  FEP %.3lf kJ/mol ( includes the variation between windows )\n"%( w_t.std(), w_f.std() )
    out += "# mean within-window std( w ): TFEP %.3lf  FEP %.3lf kJ/mol ( 1 kT = %.2lf; for a gaussian w, ESS/N = exp( -sigma^2 / kT^2 ): aim at ~1.5 kT or less )\n"%(
        numpy.nanmean( sdt ), numpy.nanmean( sdf ), qm3.utils.tfep.kt_of( TEMP ) )
    return( out )


# -------------------------------------------------------------------------------------------------
# with molecule / engines
# -------------------------------------------------------------------------------------------------

def build( engines: bool ) -> tuple:
    import  qm3
    import  qm3.engines.mmres
    mol = qm3.molecule()
    mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
    mol.psf_read( open( "oxy-cope.psf" ) )
    mol.guess_atomic_numbers()
    mol.xyz_read( open( "reac.xyz", "rt" ), replace = True )
    with open( "pmf_s.delz", "rt" ) as f:
        dz = float( f.read().strip() )
    cvs = qm3.engines.mmres.colvar_s( mol, "pmf_s.cnf", "s3/last.str", dz, True )
    e_hl = None
    if( engines ):
        import  openmm
        import  openmm.app
        import  openmm.unit
        import  qm3.engines.openmm
        import  qm3.engines.mopac
        import  io
        _psf = openmm.app.charmmpsffile.CharmmPsfFile( "oxy-cope.psf" )
        _psf.setBox( mol.boxl[0] * openmm.unit.angstrom, mol.boxl[1] * openmm.unit.angstrom, mol.boxl[2] * openmm.unit.angstrom )
        _prm = openmm.app.charmmparameterset.CharmmParameterSet( "oxy-cope.top", "oxy-cope.prm" )
        _sys = _psf.createSystem( _prm, nonbondedMethod = openmm.app.CutoffPeriodic,
            nonbondedCutoff = 16.0 * openmm.unit.angstrom, switchDistance = 14.0 * openmm.unit.angstrom, rigidWater = False )
        sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
        with open( "sele_QM.pk", "rb" ) as f:
            sqm[pickle.load( f )] = True
        smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
        with open( "sele_MM.pk", "rb" ) as f:
            smm[pickle.load( f )] = True
        mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
        e_hl = qm3.engines.mopac.run( mol, "AM1", 0, 1, sel_QM = sqm, sel_MM = smm )
        e_hl.img_init( mol )
    return( ( mol, cvs, e_hl ) )


def stage_fit():
    dat = qm3.utils.tfep.dataset.load( DAT )
    tr, mp, evl, ym, lj = split_fit( dat )
    # s must be preserved by the map
    mol, cvs, _ = build( False )
    err = 0.0
    for i in range( min( 100, len( evl ) ) ):
        mol.coor[evl.idx] = ym[i]
        err = max( err, abs( cvs.value( mol )[0] - evl.s[i] ) )
    print( "max |s( M( x ) ) - s( x )| = %.2e"%err )
    numpy.savez( MAPNPZ, tr = tr, ym = ym, logj = lj )
    with open( MAPPK, "wb" ) as f:
        pickle.dump( mp, f )


def stage_explain():
    """ no engines: how much of the within-window fluctuation of dU = E_HL - E_LL is explained by the solute coordinates """
    dat = qm3.utils.tfep.dataset.load( DAT )
    tr = split( dat )
    trn = dat.subset( tr )
    evl = dat.subset( ~tr )
    mp = qm3.utils.tfep.gauss_map( free = dat.idx, frame = pick_frame( dat ), degree = DEGREE, shrink = SHRINK, ridge = RIDGE,
            temperature = TEMP, vmax = VMAX, use_d = USE_D, fridge = FRIDGE, emax = EMAX, use_fixed = USE_FIXED )
    mp.fit( trn )
    def centred( dd ):
        qq = mp._local( dd )[0]
        du = dd.e_hl - dd.e_ll
        for w in numpy.unique( dd.win ):
            m = dd.win == w
            qq[m] -= qq[m].mean( axis = 0 )
            du[m] -= du[m].mean()
        return( ( qq, du ) )
    qt, dt = centred( trn )
    qe, de = centred( evl )
    sig = qt.std( axis = 0 )
    sig[sig < 1.e-8] = 1.0
    zt = qt / sig
    ze = qe / sig
    iu = numpy.triu_indices( zt.shape[1] )
    def quad( z ):
        return( ( z[:,:,None] * z[:,None,:] )[:,iu[0],iu[1]] )
    qa = quad( zt )
    qm = qa.mean( axis = 0 )
    qs = qa.std( axis = 0 )
    qs[qs < 1.e-12] = 1.0
    ft = numpy.hstack( ( zt, ( qa - qm ) / qs ) )
    fe = numpy.hstack( ( ze, ( quad( ze ) - qm ) / qs ) )
    for nom, xt, xe in ( ( "linear   ", zt, ze ), ( "lin + quad", ft, fe ) ):
        for lam in ( 1.e-3, 1.e-1, 1.0, 10.0 ):
            a = numpy.linalg.solve( numpy.dot( xt.T, xt ) + lam * len( dt ) * numpy.eye( xt.shape[1] ), numpy.dot( xt.T, dt - dt.mean() ) )
            res = ( de - de.mean() ) - numpy.dot( xe, a )
            print( "%s ridge %7.3lf: within-window std( dU ) = %.2lf kJ/mol  ->  residual %.2lf kJ/mol   R2 ( evaluation ) = %.3lf"%(
                nom, lam, de.std(), res.std(), 1.0 - res.var() / de.var() ) )
    print( "( for a gaussian w, ESS/N = exp( -sigma^2 / kT^2 ): the residual should get down to ~1.5 kT = %.1lf kJ/mol, ~2 kT = %.1lf is marginal )"%(
        1.5 * qm3.utils.tfep.kt_of( TEMP ), 2.0 * qm3.utils.tfep.kt_of( TEMP ) ) )


def stage_hl():
    import  qm3.utils._dcd
    dat = qm3.utils.tfep.dataset.load( DAT )
    with numpy.load( MAPNPZ ) as f:
        tr, ym = f["tr"], f["ym"]
    evl = dat.subset( ~tr )
    mol, cvs, e_hl = build( True )
    e = numpy.full( len( evl ), numpy.nan )
    if( os.path.isfile( HLNPZ ) ):
        with numpy.load( HLNPZ ) as f:
            if( f["e"].shape[0] == e.shape[0] ):
                e = f["e"].copy()
    need = {}
    for i in range( len( evl ) ):
        if( numpy.isnan( e[i] ) ):
            need.setdefault( int( evl.win[i] ), {} )[int( evl.step[i] )] = i
    ndo = 0
    first = True
    dcd = qm3.utils._dcd.dcd()
    for w in sorted( need ):
        dcd.open_read( f"s3/dcd.{w:03d}" )
        k = 0
        todo = len( need[w] )
        while( todo > 0 and dcd.next( mol.coor ) ):
            if( k in need[w] ):
                i = need[w][k]
                if( first ):
                    # the dcd frame must reproduce the stored HL energy
                    mol.get_func(); e_hl.get_func( mol )
                    print( "check ( window %d, frame %d ): E_HL stored %.4lf  recomputed %.4lf"%( w, k, evl.e_hl[i], mol.func ) )
                    if( abs( mol.func - evl.e_hl[i] ) > 0.05 ):
                        raise ValueError( "dcd frame does not reproduce data.npz: wrong dcd / step order / engines" )
                    first = False
                mol.coor[evl.idx] = ym[i]
                mol.get_func()
                e_hl.get_func( mol )
                e[i] = mol.func
                todo -= 1
                ndo += 1
                if( ndo % 10 == 0 ):
                    numpy.savez( HLNPZ, e = e )
                    print( "[hl] %d done, %d left"%( ndo, int( numpy.isnan( e ).sum() ) ) )
            k += 1
        dcd.close()
        if( todo > 0 ):
            raise RuntimeError( f"window {w}: reached end of dcd with {todo} requested frames missing" )
    numpy.savez( HLNPZ, e = e )
    print( "[hl] finished: %d evaluations"%ndo )


def stage_fes():
    dat = qm3.utils.tfep.dataset.load( DAT )
    with numpy.load( MAPNPZ ) as f:
        tr, lj = f["tr"], f["logj"]
    with numpy.load( HLNPZ ) as f:
        e = f["e"]
    out = report( dat.subset( ~tr ), e, lj )
    print( out )
    with open( OUT, "wt" ) as f:
        f.write( out )


if( __name__ == "__main__" ):
    if( len( sys.argv ) != 2 or sys.argv[1] not in ( "fit", "hl", "fes", "explain" ) ):
        print( __doc__ )
        sys.exit( 1 )
    { "fit": stage_fit, "hl": stage_hl, "fes": stage_fes, "explain": stage_explain }[sys.argv[1]]()
