import  sys
import  math
import  numpy
import  typing
import  qm3.data
import  qm3.actions.dynamics
import  qm3.engines.hyperplane


"""
Dinamica molecular sobre el hiperplano de transicion ( q( x ) = 0 ) SIN potencial de restraint:
la superficie se impone como un constraint holonomico, resuelto con RATTLE ( velocity-Verlet ).
Solo se necesita dq/dx, que depende de unos pocos atomos y de unas pocas CVs.

Ensemble muestreado
-------------------
La dinamica con constraint duro NO muestrea el ensemble condicional e^-bV d( q ) del articulo:
lo muestrea multiplicado por |dq| ( factor Z^1/2 de blue-moon, Ciccotti et al. 2005 ).
Para obtener promedios sobre el ensemble condicional ( "el TS ensemble" ):

    < A >_TS = < A / |dq| > / < 1 / |dq| >

|dq| se escribe en el log ( columna 'grad_q' ) y se devuelve en las estructuras que se guarden en
current_step ( hts.evaluate( mol ).gnorm ). Si q es casi lineal en las coordenadas ponderadas por
masa, |dq| es casi constante y la correccion es pequena; conviene comprobarlo.

Termostato: CSVR ( escalado uniforme de velocidades: preserva dq.v = 0 ) o NVE ( temperature = None,
util para disparar trayectorias desde el TS ).
"""


def project_to_plane( mol: object, hts: qm3.engines.hyperplane.colvar_hts,
        tol: typing.Optional[float] = 1.e-10,
        max_iter: typing.Optional[int] = 100 ) -> float:
    """
    lleva las coordenadas a q = 0 con el desplazamiento de minima norma ponderada por masa
    ( Newton a lo largo de W.dq ); devuelve q final
    """
    for it in range( max_iter ):
        s = hts.evaluate( mol )
        if( abs( s.q ) < tol ):
            break
        mol.coor[hts.indx] -= ( s.q / ( s.gnorm * s.gnorm ) * hts.mass * s.gq ).reshape( ( len( hts.indx ), 3 ) )
    return( s.q )


def project_velocities( mol: object, hts: qm3.engines.hyperplane.colvar_hts,
        s: typing.Optional[object] = None ):
    """ elimina de las velocidades la componente a lo largo de dq ( dq . v = 0 ) """
    if( s is None ):
        s = hts.evaluate( mol )
    v = mol.velo[hts.indx].ravel()
    v = v - float( s.gq @ v ) / ( s.gnorm * s.gnorm ) * hts.mass * s.gq
    mol.velo[hts.indx] = v.reshape( ( len( hts.indx ), 3 ) )


def _remove_com( mol: object ):
    # velocidad del centro de masas real ( las CVs son invariantes a traslaciones: sum dq = 0,
    # asi que esto no rompe dq . v = 0, a diferencia de la proyeccion sqrt( m ) de dynamics.py )
    m = mol.mass * mol.actv
    mol.velo -= numpy.sum( m * mol.velo, axis = 0 ) / numpy.sum( m )


def constrained_verlet( mol: object, hts: qm3.engines.hyperplane.colvar_hts,
        step_size: typing.Optional[float] = 0.001,
        temperature: typing.Optional[float] = 300.0,
        temperature_coupling: typing.Optional[float] = 0.1,
        print_frequency: typing.Optional[int] = 100,
        step_number: typing.Optional[int] = 1000,
        tolerance: typing.Optional[float] = 1.e-9,
        log_file: typing.Optional[typing.IO] = sys.stdout,
        current_step: typing.Optional[typing.Callable] = qm3.actions.dynamics.fake_cs ):
    """
    RATTLE ( velocity-Verlet con el constraint q( x ) = 0 ) + CSVR [J. Chem. Phys. v126, p14101 (2007)]

    temperature = None  =>  NVE
    mol.engines NO debe contener ningun restraint sobre q ( ni qm3.engines.hyperplane.bias )
    """
    for itm in mol.engines:
        if( isinstance( mol.engines[itm], qm3.engines.hyperplane.bias ) ):
            raise ValueError( "constrained_verlet: quita el engine 'bias' ( el constraint ya impone q = 0 )" )
    nvt = temperature is not None
    ndeg = 3 * int( mol.actv.sum() )
    rcom = ( mol.actv.sum() == mol.natm )
    if( rcom ):
        ndeg -= 3
    ndeg -= 1                                                  # el constraint
    log_file.write( "---------------------------------------- Dynamics: RATTLE on q(x)=0 (%s)\n\n"%( "CSVR-NVT" if nvt else "NVE" ) )
    log_file.write( "Degrees of Freedom: %20ld [constraint%s]\n"%( ndeg, ", COM removed" if rcom else "" ) )
    log_file.write( "Step Size:          %20.10lg (ps)\n"%( step_size ) )
    if( nvt ):
        log_file.write( "Temperature:        %20.10lg (K)\n"%( temperature ) )
        log_file.write( "Temp. coupling:     %20.10lg (ps)\n"%( temperature_coupling ) )
    log_file.write( "Step Number:        %20d\n"%( step_number ) )
    log_file.write( "Print Frequency:    %20d\n"%( print_frequency ) )
    log_file.write( "\n%20s%20s%20s%20s%20s%20s%20s\n"%( "Time (ps)", "Potential (kJ/mol)", "Kinetic (kJ/mol)",
        "Total (kJ/mol)", "Temperature (K)", "q", "grad_q" ) )
    log_file.write( 140 * "-" + "\n" )
    if( nvt ):
        c0 = math.exp( - step_size / temperature_coupling )
        c1 = 5.e-4 * qm3.data.KB * temperature * qm3.data.NA * ( 1.0 - c0 )
        c2 = 2.0 * math.sqrt( c0 * c1 )
    fc = step_size
    fv = 0.5 * step_size
    fa = fc * fv
    # ---- estado inicial: sobre el plano y con velocidades tangentes
    q0 = project_to_plane( mol, hts, tolerance )
    if( abs( q0 ) > 1.e-6 ):
        raise RuntimeError( "constrained_verlet: cannot reach q = 0 ( q = %.3le )"%( q0 ) )
    if( not hasattr( mol, "velo" ) ):
        if( not nvt ):
            raise ValueError( "constrained_verlet: NVE needs mol.velo" )
        proj = numpy.zeros( ( mol.natm, 1 ) )
        qm3.actions.dynamics.assign_velocities( mol, temperature, proj, ndeg )
    if( rcom ):
        _remove_com( mol )
    project_velocities( mol, hts )
    temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
    if( nvt and temp > 0.0 ):
        mol.velo *= math.sqrt( temperature / temp )
        temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
    mol.get_grad()
    cacc = - mol.grad / mol.mass * 100.0
    s = hts.evaluate( mol )
    xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
    xavr = xtmp.copy()
    xrms = numpy.square( xtmp )
    time = 0.0
    qmax = abs( s.q )
    log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.3le%20.8lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3], s.q, s.gnorm ) )
    for istp in range( 1, step_number + 1 ):
        time += step_size
        # ---- posiciones ( SHAKE sobre q ): x' = x + dt v + dt^2/2 ( a - mu d ),  d = W dq( x_n )
        dold = hts.mass * s.gq
        mol.coor += fc * mol.velo + fa * cacc
        loc0 = mol.coor[hts.indx].copy()
        mu   = 0.0
        for it in range( 100 ):
            mol.coor[hts.indx] = loc0 - ( 0.5 * fc * fc * mu * dold ).reshape( loc0.shape )
            sn = hts.evaluate( mol )
            if( abs( sn.q ) < tolerance ):
                break
            mu += sn.q / ( 0.5 * fc * fc * float( sn.gq @ dold ) )
        else:
            raise RuntimeError( "constrained_verlet: SHAKE did not converge at step %d ( q = %.3le )"%( istp, sn.q ) )
        # ---- primera mitad de la velocidad
        mol.velo += fv * cacc
        mol.velo[hts.indx] -= ( fv * mu * dold ).reshape( loc0.shape )
        mol.get_grad()
        cacc = - mol.grad / mol.mass * 100.0
        mol.velo += fv * cacc
        # ---- RATTLE: dq . v = 0 en x_{n+1}
        s = hts.evaluate( mol )
        project_velocities( mol, hts, s )
        if( nvt ):
            temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
            rr  = numpy.random.normal( 0.0, 1.0, ndeg )
            scv = math.sqrt( c0 + c1 / kine * ( rr[0] * rr[0] + numpy.sum( numpy.square( rr[1:] ) ) ) + c2 / math.sqrt( kine ) * rr[0] )
            mol.velo *= scv
            if( rcom ):
                _remove_com( mol )
            project_velocities( mol, hts, s )
        temp, kine = qm3.actions.dynamics.current_temperature( mol, ndeg )
        xtmp = numpy.array( [ mol.func, kine, mol.func + kine, temp ], dtype=numpy.float64 )
        xavr += xtmp
        xrms += numpy.square( xtmp )
        qmax = max( qmax, abs( s.q ) )
        if( istp % print_frequency == 0 ):
            log_file.write( "%20.5lf%20.5lf%20.5lf%20.5lf%20.5lf%20.3le%20.8lf\n"%( time, xtmp[0], xtmp[1], xtmp[2], xtmp[3], s.q, s.gnorm ) )
        current_step( mol, istp )
    xavr /= step_number + 1
    xrms /= step_number + 1
    xrms = numpy.sqrt( numpy.fabs( xrms - xavr * xavr ) )
    log_file.write( 140 * "-" + "\n" )
    log_file.write( "%-20s"%( "Averages:" ) + "".join( [ "%20.5lf"%( i ) for i in xavr ] ) + "\n" )
    log_file.write( "%-20s"%( "RMS Deviations:" ) + "".join( [ "%20.5lf"%( i ) for i in xrms ] ) + "\n" )
    log_file.write( "%-20s%20.3le\n"%( "max |q|:", qmax ) )
    log_file.write( 140 * "-" + "\n" )
