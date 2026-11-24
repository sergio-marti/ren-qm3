#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mmres
import  qm3.engines.hyperplane
import  qm3.actions.dynamics
import  qm3.actions.hts_dynamics
import  qm3.utils.parallel
import  qm3.utils._dcd
import  qm3.utils.hessian
import  pickle
import  sys

"""
Hiperplano de transicion sobre las distancias de una colvar_s ( oxy-Cope en agua )

  python hts.py init           # 1 proceso: Hessiano del TS -> plano inicial ( hts_init.json )
  python hts.py optimize       # NWALK procesos ( client_fsi ): mitad en reac.xyz, mitad en prod.xyz
  python hts.py final          # tabla de Cz / Cn por ps ( para elegir la ventana de promediado )
  python hts.py final 5.0 10.0 # promedia esa ventana y escribe hts.json
  python hts.py produce        # dinamica en q = 0 SIN ningun restraint ( constraint RATTLE )

El plano inicial se calcula UNA vez ( init ) y no en cada walker: todos los walkers deben partir del MISMO hiperplano.

Opcional ( variables de entorno ): HTS_NWALK, HTS_KMB, HTS_TWARM, HTS_TTOT, HTS_TCKPT,
HTS_INIT = ts ( por defecto, plano del modo imaginario del TS ) | rp ( plano desde reac.xyz y prod.xyz ).
( para continuar desde un plano ya optimizado: python hts.py final 5 10 ; HTS_INIT=hts.json python hts.py optimize ).
"""

STAGE = sys.argv[1] if len( sys.argv ) > 1 else "optimize"
NWALK = int( os.environ.get( "HTS_NWALK", 10 ) )         # mitad en reactivo, mitad en producto
KMB  = float( os.environ.get( "HTS_KMB", 12000.0 ) )     # ps^-2: DEBE superar kmin ( lo imprime init ), mejor >= 1.5 kmin
DMAX = 3.0                                               # amu^1/2 Ang ( el articulo usa 3 para la IPL; vigilar wall_report )
DT, NSYNC = 0.0005, 10                                   # ps ; un update del plano cada NSYNC pasos de MD ( 5 fs )
T_WARM = float( os.environ.get( "HTS_TWARM", 1.0 ) )     # ps: equilibrado con el plano FIJO ( los walkers llegan al plano desde R / P )
T_TOT  = float( os.environ.get( "HTS_TTOT", 10.0 ) )     # ps de optimizacion tras el equilibrado
T_CKPT = float( os.environ.get( "HTS_TCKPT", 1.0 ) )     # ps entre puntos de control ( hts_last.json, hts_hist.pk )


if( STAGE == "final" ):
    # no necesita construir el sistema
    with open( "hts_hist.pk", "rb" ) as f:
        hist = pickle.load( f )
    t  = numpy.array( [ h[0] for h in hist ] )
    cz = numpy.array( [ h[3] for h in hist ] )
    cn = numpy.array( [ h[4] for h in hist ] )
    print( "%10s %10s %10s %10s %10s %12s"%( "t (ps)", "<Cz>", "sd Cz", "<Cn> (deg)", "sd Cn", "dCn/dt" ) )
    for a in numpy.arange( numpy.floor( t[0] ), t[-1], 1.0 ):
        s = ( t > a ) & ( t <= a + 1.0 )
        if( s.sum() > 5 ):
            print( "%4.0f-%-5.0f %10.3f %10.3f %10.2f %10.2f %12.2f"%( a, a + 1.0, cz[s].mean(), cz[s].std(), cn[s].mean(), cn[s].std(),
                numpy.polyfit( t[s], cn[s], 1 )[0] ) )
    # margen de K frente al umbral K_min, estimado de la propia dinamica:  K_eff = K - K_min = kT / var( q )
    import glob
    wl = sorted( glob.glob( "hts_walker.*.log" ) )
    if( wl ):
        with open( wl[0] ) as f:
            first = f.readline()
        if( first.startswith( "#K" ) ):
            Kw, kTw = float( first.split()[1] ), float( first.split()[3] )
        else:                       # logs antiguos: pasar la K usada en HTS_KMB
            Kw, kTw = float( os.environ.get( "HTS_KMB", "nan" ) ), qm3.engines.hyperplane.kT_int( 300.0 )
        if( Kw == Kw ):
            W = [ numpy.loadtxt( f ) for f in wl ]
            print( "\nmargen de K ( K = %.0f ps^-2 ): K_eff = kT / var( q ), K_min ~ K - K_eff, margen = K / K_min"%( Kw ) )
            print( "%10s %10s %12s %12s %10s"%( "t (ps)", "sd( q )", "K_eff", "K_min est.", "margen" ) )
            for a in numpy.arange( numpy.floor( W[0][0,0] ), W[0][-1,0], 1.0 ):
                q = numpy.concatenate( [ w[( w[:,0] > a ) & ( w[:,0] <= a + 1.0 ),1] for w in W ] )
                if( len( q ) > 20 ):
                    ke = kTw / q.var()
                    km = Kw - ke
                    print( "%4.0f-%-5.0f %10.3f %12.0f %12.0f %10.2f %s"%( a, a + 1.0, q.std(), ke, km, Kw / km if km > 0 else float( "inf" ),
                        "  <-- margen < 1.5: subir K" if ( km > 0 and Kw / km < 1.5 ) else "" ) )
    if( len( sys.argv ) > 3 ):
        z, n, k, t0, t1 = qm3.engines.hyperplane.final_from_files( "hts_hist.pk", "hts_last.json",
                float( sys.argv[2] ), "hts.json", float( sys.argv[3] ) )
        print( "hts.json escrito con %d puntos ( t = %.2f - %.2f ps )"%( k, t0, t1 ) )
        print( "  z =", z, "\n  n =", n )
    sys.exit( 0 )


opar = qm3.utils.parallel.client_fsi( NWALK ) if STAGE == "optimize" else None
node = opar.node if opar is not None else 0
qm3.actions.dynamics.seed()

if( STAGE == "optimize" ):
    start = "reac.xyz" if node < NWALK // 2 else "prod.xyz"
else:
    start = "saddle.xyz"

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
mol.xyz_read( open( start, "rt" ), replace = True )

_psf = openmm.app.charmmpsffile.CharmmPsfFile( "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( "oxy-cope.top", "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

mol.engines["qm"] = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
mol.engines["qm"].img_init( mol )

hts = qm3.engines.hyperplane.colvar_hts( mol, "pmf_s.cnf" )

if( STAGE == "init" ):
    reac = mol.copy()
    reac.xyz_read( open( "reac.xyz", "rt" ), replace = True )
    mol.set_active( sqm )
    hess = qm3.utils.hessian.numerical( mol )
    mol.set_active()
    info = hts.initial_guess_ts( mol, hess, sele = sqm, react = reac )
    print( info ) # overlap ~ 1: las distancias expresan el modo imaginario; nneg debe ser 1; si overlap es bajo, faltan CVs
    print( "K minima para confinar al walker en el plano: kmin = %.0f ps^-2 ( usa KMB >= %.0f ); KMB actual = %.0f"%(
        info["kmin"], 1.5 * info["kmin"], KMB ) )
    with open( "hts_init.kmin", "wt" ) as f:
        f.write( "%.6f\n"%( info["kmin"] ) )
    hts.save( "hts_init.json" )
    md = hts.ts_mode( mol, hess, sele = sqm )
    if( os.path.isfile( "prod.xyz" ) ):
        # plano alternativo desde reactivo y producto ( eqs. 11-14 ) y comparacion con el plano del modo imaginario del TS
        prod = mol.copy()
        prod.xyz_read( open( "prod.xyz", "rt" ), replace = True )
        hrp = qm3.engines.hyperplane.colvar_hts( mol, "pmf_s.cnf" )
        hrp.initial_guess( [ reac ], [ prod ] )
        q_ts = hrp.value( mol )[0]                       # lo lejos que queda el TS del plano que pasa por el punto medio R/P
        n2 = hrp.n / hts.norm_M( hrp.n )                 # n del plano R/P, con longitud unidad bajo la metrica M del TS
        cosang = min( 1.0, max( -1.0, float( hts.n @ hts.Minv @ n2 ) ) )
        # z en el plano es un grado de libertad de gauge que casi no se mueve ( gamma_p = 1000 ps ) y el muro dmax se mide
        # respecto a el: se centra en el TS ( se conservan n y M del plano R/P ) para que el muro no actue sobre el ensemble del TS
        hrp.set_state( hrp.get_info( mol )[0], hrp.n, hrp.M )
        hrp.save( "hts_init_rp.json" )
        print( "plano R/P guardado en hts_init_rp.json ( z centrado en el TS ): angulo con el plano del TS = %.1f grados ; q( TS ) en el plano R/P-punto medio = %+.3f amu^1/2 Ang"%(
            numpy.degrees( numpy.arccos( cosang ) ), q_ts ) )
        kmin_rp = hrp.k_min( mol, md )
        print( "  kmin con el plano R/P = %.0f ps^-2 ( con el plano del TS: %.0f ): con HTS_INIT=rp usa KMB >= %.0f"%( kmin_rp, info["kmin"], 1.5 * kmin_rp ) )
        with open( "hts_init_rp.kmin", "wt" ) as f:
            f.write( "%.6f\n"%( kmin_rp ) )

elif( STAGE == "optimize" ):
    _ini = os.environ.get( "HTS_INIT", "ts" )          # ts | rp | un .json ( p.ej. el plano de una optimizacion anterior )
    hts.load( _ini if _ini.endswith( ".json" ) else ( "hts_init_rp.json" if _ini == "rp" else "hts_init.json" ) )   # el MISMO plano en todos los walkers
    _kf = "hts_init_rp.kmin" if _ini == "rp" else ( "hts_init.kmin" if _ini == "ts" else None )     # kmin del plano que se carga
    if( node == 0 and _kf is not None and os.path.isfile( _kf ) ):
        kmin = float( open( _kf ).read() )
        if( KMB < 1.5 * kmin ):
            print( "AVISO: KMB = %.0f < 1.5 * kmin = %.0f: el sesgo no confina bien a los walkers en el plano ( ver init )"%( KMB, 1.5 * kmin ) )
    bia = qm3.engines.hyperplane.bias( hts, KMB, dmax = DMAX )
    mol.engines["hts"] = bia
    opt = qm3.engines.hyperplane.optimizer( hts, KMB, temperature = 300., opar = opar,
            log_file = open( "hts_conv.log", "wt" ) if node == 0 else None,
            walker_log = open( "hts_walker.%03d.log"%( node ), "wt" ) )
    stp = qm3.actions.dynamics.stepper__langevin_verlet( mol, DT, 300., 50.0 )
    dcd = qm3.utils._dcd.dcd()
    dcd.open_write( "hts_conv.%03d.dcd"%( node ), mol.natm )
    for it in range( int( round( T_WARM / DT ) ) ):          # plano fijo: sin collect ni update
        stp.integrate( mol )
    bia.reset_wall()
    nckpt = int( round( T_CKPT / ( DT * NSYNC ) ) )
    for it in range( 1, int( round( T_TOT / ( DT * NSYNC ) ) ) + 1 ):
        for k in range( NSYNC ):
            stp.integrate( mol )
            opt.collect( bia.sample )
        opt.update( DT * NSYNC )
        dcd.append( mol.coor )
        if( it % nckpt == 0 and node == 0 ):
            hts.save( "hts_last.json" )
            opt.save_history( "hts_hist.pk" )
    dcd.close()
    f, d, dm = bia.wall_report()
    print( node, "dmax = %-5s -> fraccion de pasos con muro activo = %.3f ; <d> = %.3f ; max d = %.3f"%( DMAX, f, d, dm ) )
    if( node == 0 ):
        hts.save( "hts_last.json" )
        opt.save_history( "hts_hist.pk" )
    opar.barrier()
    opar.stop()

else:
    mol.dcd = qm3.utils._dcd.dcd()
    mol.dcd.open_write( "hts_md.dcd", mol.natm )
    hts.load( "hts.json" )
    print( "q en saddle.xyz antes de proyectar al plano = %.4f ( amu^1/2 Ang )"%( hts.value( mol )[0] ) )
    flog = open( "hts_md.d.log", "wt" )
    flog.write( "#step  q  d(in-plane)\n" )
    def keep( m, i ):
        if( i % 100 == 0 ):
            m.dcd.append( m.coor )
            flog.write( "%d %.3e %.4f\n"%( i, hts.value( m )[0], hts.inplane( m ) ) )   # d: hacia donde se desplaza el sistema en el plano
            flog.flush()
    qm3.actions.hts_dynamics.constrained_verlet( mol, hts, step_size = DT, temperature = 300., temperature_coupling = 0.1,
        step_number = 20_000, print_frequency = 100, current_step = keep )
    mol.dcd.close()
    flog.close()
