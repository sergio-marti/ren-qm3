import  math
import  numpy
import  typing
import  qm3.data
import  qm3.utils.hessian
import  os


def current_step( self: object, step: int ):
    with open( "rpi.xyz", "wt" ) as f:
        f.write( "%d\n\n"%( self.size ) )
        for i in range( self.half + 1 ):
            i_cc = i * self.dime
            for j in range( self.dime ):
                f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%(
                    qm3.data.symbol[self.mole.anum[self.sele[j]]],
                    self.coor[i_cc+j,0], self.coor[i_cc+j,1], self.coor[i_cc+j,2] ) )



class instanton( object ):
    """
    U = 1 over P sum from{ i=1 } to { P }{ V_i left( x_{i,1}, dotslow ,x_{i,3N} right)}+
    sum from{ i=1 } to { P }{ sum from{ j=1 } to { 3N }{
    k color gray { left lbrace  {2 P %pi^2 k_B^2 T^2 } over {h^2} 10^-26 right rbrace }
    m_{j/3} left( x_{i,j} - x_{i-1,j} right)^2  } }
    newline
    {partial U} over {partial x_{i,k}} = 1 over P {partial V_i} over {partial x_{i,k}} left( x_{i,1}, dotslow
    ,x_{i,3N} right)+2 k m_{ k/3 }left( 2x_{i,k} - x_{i-1,k} - x_{i+1,k} right)
    newline
    {partial^2 U} over {partial x_{i,k} partial x_{j,l}} =
    %delta_{j=i,l,k} over P 
    {partial^2 V_i} over {partial x_{i,k} partial x_{j,l}}
    left(x_{i,1}, dotslow ,x_{i,3N} right) +4 k m_{ k/3 } %delta_{j=i,l=k} -2 k m_{ k/3 } %delta_{j=i+1,l=k}
    -2 k m_{ k/3 } %delta_{j=i-1,l=k}

    J. Phys. Chem. Lett. v7, p4374 (2016) [10.1021/acs.jpclett.6b02115]
    J. Chem. Phys. v134, p184107 (2011) [10.1063/1.3587240]
    J. Chem. Phys. v148, p102334 (2018) [10.1063/1.5007180]
    """
    def __init__( self, mol: object,
            mol_hess: typing.Callable,
            nbeads: typing.Optional[int] = 64,
            temperature: typing.Optional[float] = 300.0 ):
        self.mole = mol
        self.CHes = mol_hess
        self.sele = numpy.flatnonzero( mol.actv.ravel() )
        self.temp = temperature
        self.bead = nbeads
        self.half = self.bead // 2
        self.kumb = 2.0 * self.bead * math.pow( self.temp * qm3.data.KB * math.pi / qm3.data.H, 2.0 ) * 1.0e-26
        print( "[RPI] bead:", self.bead, self.half + 1 )
        print( "[RPI] temp: %.2lf _K"%( self.temp ) )
        print( "[RPI] kumb: %.2lf _kJ/(mol A^2)"%( self.kumb ) )
        self.dime = len( self.sele )
        self.size = self.dime * ( self.half + 1 )
        self.actv = numpy.ones( ( self.size, 1 ), dtype=numpy.bool_ )
        self.mass = numpy.zeros( ( self.size, 1 ), dtype=numpy.float64 )
        for i in range( self.half + 1 ):
            for j in range( self.dime ):
                self.mass[i*self.dime+j] = self.mole.mass[self.sele[j]]


    @staticmethod
    def __rotations( sele: numpy.array, mass: numpy.array, coor: numpy.array, symm: float, temp: float ) -> float:
        cte = ( 8.0 * math.pi * math.pi * qm3.data.KB * temp ) / ( qm3.data.H * qm3.data.H * qm3.data.NA ) * 1.0e-23
        cen = numpy.sum( mass[sele] * coor[sele,:], axis = 0 ) / numpy.sum( mass[sele] )
        xx = 0.0; xy = 0.0; xz = 0.0; yy = 0.0; yz = 0.0; zz = 0.0
        for i in sele:
            xx += mass[i] * ( coor[i,0] - cen[0] ) * ( coor[i,0] - cen[0] )
            xy += mass[i] * ( coor[i,0] - cen[0] ) * ( coor[i,1] - cen[1] )
            xz += mass[i] * ( coor[i,0] - cen[0] ) * ( coor[i,2] - cen[2] )
            yy += mass[i] * ( coor[i,1] - cen[1] ) * ( coor[i,1] - cen[1] )
            yz += mass[i] * ( coor[i,1] - cen[1] ) * ( coor[i,2] - cen[2] )
            zz += mass[i] * ( coor[i,2] - cen[2] ) * ( coor[i,2] - cen[2] )
        val, vec = numpy.linalg.eigh( numpy.array( [ yy+zz, -xy, -xz, -xy, xx+zz, -yz, -xz, -yz, xx+yy ] ).reshape( ( 3, 3 ) ) )
        return( math.log( math.sqrt( math.pi * cte * cte * cte * val[0] * val[1] * val[2] ) / symm ) )


    def calc_TST( self,
            r_coor: numpy.array, r_func: float, r_hess: numpy.array,
            t_coor: numpy.array, t_func: float, t_hess: numpy.array,
            r_symm: typing.Optional[float] = 1.0,
            t_symm: typing.Optional[float] = 1.0 ):
        """
    Q_rot = 1 over %sigma left( %pi left( {8 %pi^2 k_B T 10^-23} over {h^2 N_A} right)^3 det I right)^{ 1 over 2 }
    ~~~~~~~
    size 10 { I =  sum from{j=1} to{N}{ m_j left[ left( {vec{r}}_{j} cdot {vec{r}}_{j} right) I_3 - {vec{r}}_{j} times {vec{r}}_{j} right] } }
    ~~~~~~~
    size 10 { {vec{r}}_{j} = left( x_{j},y_{j},z_{j} right) - left( x_{CM},y_{CM},z_{CM} right) }
    newline
    Q_vib =  prod from{k=1} to{3N-6/7} { 1 over {2 sinh left( 1 over 2 {h  %ípsilon_k 100 c} over {k_B T} right) } }
    ~~~~~~~
    k_TST = {k_B T} over h { Q_rot^{%Ux2021 } · Q_vib^{%Ux2021 } } over { Q_rot^{R } · Q_vib^{R } } e^{ - {{V^{ %Ux2021 } - V^R} over {k_B T}} 10^3 }
        """
        # activation potential energy
        efunc = - ( t_func - r_func ) * 1000.0 / ( self.temp * qm3.data.KB * qm3.data.NA )
        print( "[TST] dfunc: %20.10le (%.2lf _kJ/mol)"%( efunc, t_func - r_func ) )
        # rotational partition function
        rQR = self.__rotations( self.sele, self.mole.mass, r_coor, r_symm, self.temp )
        print( "[TST] l_rQR: %20.10le"%( rQR ) )
        tQR = self.__rotations( self.sele, self.mole.mass, t_coor, t_symm, self.temp )
        print( "[TST] l_tQR: %20.10le"%( tQR ) )
        # vibrational partition function
        cte = 100.0 * qm3.data.C * qm3.data.H / ( qm3.data.KB * self.temp )
        self.mole.coor = r_coor
        frq = qm3.utils.hessian.frequencies( self.mole, r_hess )[0]
        #rQV = 0.0
        #for f in frq[6:]:
        #    rQV -= math.log( 2.0 * math.sinh( f * cte * 0.5 ) )
        rQV = - numpy.sum( numpy.log( 2.0 * numpy.sinh( frq[6:] * cte * 0.5 ) ) )
        print( "[TST] rfreq: " + ", ".join( [ "%.1lf"%( math.fabs( i ) ) for i in frq[0:7] ] ) + " _cm^-1" )
        print( "[TST] l_rQV: %20.10le"%( rQV ) )
        self.mole.coor = t_coor
        frq, vec = qm3.utils.hessian.frequencies( self.mole, t_hess )
        self.tst_mode = vec[:,0]
        self.tst_mode /= numpy.linalg.norm( self.tst_mode )
        self.tst_mode.shape = ( self.dime, 3 )
        tQV = - numpy.sum( numpy.log( 2.0 * numpy.sinh( frq[7:] * cte * 0.5 ) ) )
        print( "[TST] tfreq: " + ", ".join( [ "%.1lf"%( math.fabs( i ) ) for i in frq[0:9] ] ) + " _cm^-1" )
        print( "[RPI]  T_co: %.2lf _K"%( math.fabs( frq[0] ) * 100.0 * qm3.data.C * qm3.data.H / ( 2.0 * math.pi * qm3.data.KB ) ) )
        print( "[TST] l_tQV: %20.10le"%( tQV ) )
        # kinetic constant
        self.k_TST = qm3.data.KB * self.temp / qm3.data.H * math.exp( tQV + tQR - rQV - rQR + efunc )
        print( "[TST] k_cin: %20.10le _1/s"%( self.k_TST  ) )


    def setup( self, step_size: typing.Optional[float] = 0.3 ):
        dsp       = 2 * step_size / self.half
        self.coor = numpy.zeros( ( self.size, 3 ), dtype=numpy.float64 )
        for i in range( self.half + 1 ):
            for j in range( self.dime ):
                self.coor[i*self.dime+j] = self.mole.coor[self.sele[j]] + ( i * dsp - step_size ) * self.tst_mode[j]


    def get_grad( self ):
        self.ener = numpy.zeros( self.half + 1, dtype=numpy.float64 )
        self.func = 0.0
        self.grad = numpy.zeros( ( self.size, 3 ), dtype=numpy.float64 )
        for i in range( self.half + 1 ):
            i_cc = i * self.dime
            if( i == 0 ):
                scal = 1.0
                i_mm = ( i + 1 ) * self.dime
                i_pp = ( i + 1 ) * self.dime
            elif( i == self.half ):
                scal = 1.0
                i_mm = ( i - 1 ) * self.dime
                i_pp = ( i - 1 ) * self.dime
            else:
                scal = 2.0
                i_mm = ( i - 1 ) * self.dime
                i_pp = ( i + 1 ) * self.dime
            self.mole.coor[self.sele] = self.coor[i_cc:i_cc+self.dime]
            self.mole.get_grad()
            self.ener[i] = self.mole.func
            self.func += scal * self.mole.func / self.bead
            for j in range( self.dime ):
                self.grad[i_cc+j] = self.mole.grad[self.sele[j]] / self.bead
                kmb = self.mass[j][0] * self.kumb
                self.func += scal * kmb * numpy.sum( numpy.square( self.coor[i_cc+j] - self.coor[i_mm+j] ) )
                self.grad[i_cc+j] += 2.0 * kmb * ( 2.0 * self.coor[i_cc+j] - self.coor[i_mm+j] - self.coor[i_pp+j] )


    @staticmethod
    def get_hess( self: object, step: int, fresh: typing.Optional[int] = 1 ):
        size = self.size * 3
        dime = self.dime * 3
        self.hess = numpy.zeros( ( size, size ), dtype=numpy.float64 )
        if( step % fresh == 0 ):
            for i in range( self.half + 1 ):
                i_cc = i * self.dime
                self.mole.coor[self.sele] = self.coor[i_cc:i_cc+self.dime]
                j_cc = i * dime
                self.hess[j_cc:j_cc+dime,j_cc:j_cc+dime] = self.CHes( self.mole, step ) / self.bead
                if( i == 0 ):
                    j_mm = size - dime
                    j_pp = dime
                elif( i == self.half ):
                    j_mm = j_cc - dime
                    j_pp = 0
                else:
                    j_mm = j_cc - dime
                    j_pp = j_cc + dime
                for j in range( dime ):
                    kmb = self.mass[j//3][0] * self.kumb
                    self.hess[j_cc+j,j_cc+j] += 4.0 * kmb
                    self.hess[j_cc+j,j_mm+j] -= 2.0 * kmb
                    self.hess[j_cc+j,j_pp+j] -= 2.0 * kmb
            # split get_grad thus being called also from get_hess (on the already calculated self.mole.grad/func)
            self.get_grad()
            qm3.utils.hessian.manage( self, self.hess, dump_name = "rpi.dump" )
        else:
            self.get_grad()
            qm3.utils.hessian.manage( self, self.hess, dump_name = "rpi.dump", should_update = True )
        return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) )


    def calc_RPI( self, r_coor: numpy.array, r_func: float, r_hess: numpy.array ):
        """
Q_rot = left[ 1 over %sigma right]_R left( %pi left( {8 %pi^2 P k_B T 10^-23} over{h^2 N_A} right)^3 det I right)^{ 1 over 2 }
~~~~~
size 10 { 
I = sum from{ i=1 } to{ P }{ sum from{j=1} to{N}{ m_j left[ left( {vec{r}}_{i,j} cdot
{vec{r}}_{i,j} right) I_3 - {vec{r}}_{i,j} times {vec{r}}_{i,j} right] } } }
~~~~~
size 9{ {vec{r}}_{i,j} = left( x_{i,j},y_{i,j},z_{i,j} right) - left( x_{CM},y_{CM},z_{CM} right) }
newline
Q_vib = prod from{k=1} to{3N cdot P-6} { 1 over {2 sinh left( 1 over 2 {h lline %ípsilon_k rline 100 c } over {P k_B T} right) } }
newline
k_{RP} = {k_B T P} over h
left( {2 B %pi k_B T P 10^-23} over{h^2 N_A} right)^{1 over 2} ~~ 
{ Q_rot^{%Ux2021 } · Q_vib^{%Ux2021 } } over { Q_rot^{R } · Q_vib^{R } } e^{ - {{U^{ %Ux2021 } - V^R} over {k_B T}} 10^3 }
~~~~~~~~
size 10 { B= sum from{ i=1 } to { P }{ sum from{j=1} to{3N} { m_{j/3} left( x_{i,j} - x_{i-1,j} right)^2 } } }
        """
        # activation potential energy
        efunc = - ( self.func - r_func ) * 1000.0 / ( self.temp * qm3.data.KB * qm3.data.NA )
        print( "[RPI] dfunc: %20.10le (%.2lf _kJ/mol)"%( efunc, self.func - r_func ) )
        # rotational partition function
        rQR = self.__rotations( self.sele, self.mole.mass, r_coor, 1.0, self.temp ) + 3.0 * math.log( self.bead )
        print( "[RPI] l_rQR: %20.10le"%( rQR ) )
        cen = numpy.sum( self.mass * self.coor, axis = 0 ) / numpy.sum( self.mass )
        xx = 0.0; xy = 0.0; xz = 0.0; yy = 0.0; yz = 0.0; zz = 0.0
        for i in range( self.half + 1 ):
            i_cc = i * self.dime
            if( i == 0 ):
                scal = 1.0
            elif( i == self.half ):
                scal = 1.0
            else:
                scal = 2.0
            for j in range( self.dime ):
                xx += scal * self.mass[j] * ( self.coor[i_cc+j,0] - cen[0] ) * ( self.coor[i_cc+j,0] - cen[0] )
                xy += scal * self.mass[j] * ( self.coor[i_cc+j,0] - cen[0] ) * ( self.coor[i_cc+j,1] - cen[1] )
                xz += scal * self.mass[j] * ( self.coor[i_cc+j,0] - cen[0] ) * ( self.coor[i_cc+j,2] - cen[2] )
                yy += scal * self.mass[j] * ( self.coor[i_cc+j,1] - cen[1] ) * ( self.coor[i_cc+j,1] - cen[1] )
                yz += scal * self.mass[j] * ( self.coor[i_cc+j,1] - cen[1] ) * ( self.coor[i_cc+j,2] - cen[2] )
                zz += scal * self.mass[j] * ( self.coor[i_cc+j,2] - cen[2] ) * ( self.coor[i_cc+j,2] - cen[2] )
        val, vec = numpy.linalg.eigh( numpy.array( [ yy+zz, -xy, -xz, -xy, xx+zz, -yz, -xz, -yz, xx+yy ] ).reshape( ( 3, 3 ) ) )
        cte = ( 8.0 * math.pi * math.pi * qm3.data.KB * self.temp * self.bead ) / ( qm3.data.H * qm3.data.H * qm3.data.NA ) * 1.0e-23
        tQR = math.log( math.sqrt( math.pi * cte * cte * cte * val[0] * val[1] * val[2] ) )
        print( "[RPI] l_tQR: %20.10le"%( tQR ) )
        # vibrational partition function: collapsed reactants
        o_actv = self.actv
        o_coor = self.coor
        o_mass = self.mass
        dime = self.dime * 3
        size = self.bead * dime
        hess = numpy.zeros( ( size, size ), dtype=numpy.float64 )
        self.coor = numpy.zeros( ( self.bead * self.dime, 3 ), dtype=numpy.float64 )
        self.mass = numpy.zeros( ( self.bead * self.dime, 1 ), dtype=numpy.float64 )
        self.actv = numpy.ones( ( self.bead * self.dime, 1 ), dtype=numpy.bool_ )
        for i in range( self.bead ):
            i_cc = i * self.dime
            j_cc = i * dime
            hess[j_cc:j_cc+dime,j_cc:j_cc+dime] = r_hess / self.bead
            self.coor[i_cc:i_cc+self.dime] = r_coor
            self.mass[i_cc:i_cc+self.dime] = self.mole.mass[self.sele]
            if( i == 0 ):
                j_mm = size - dime
                j_pp = dime
            elif( i == self.bead - 1 ):
                j_mm = j_cc - dime
                j_pp = 0
            else:
                j_mm = j_cc - dime
                j_pp = j_cc + dime
            for j in range( dime ):
                kmb = self.mass[j//3][0] * self.kumb
                hess[j_cc+j,j_cc+j] += 4.0 * kmb
                hess[j_cc+j,j_mm+j] -= 2.0 * kmb
                hess[j_cc+j,j_pp+j] -= 2.0 * kmb
        frq = qm3.utils.hessian.frequencies( self, hess )[0]
        cte = 100.0 * qm3.data.C * qm3.data.H / ( self.bead * qm3.data.KB * self.temp )
        rQV = - numpy.sum( numpy.log( 2.0 * numpy.sinh( frq[6:] * cte * 0.5 ) ) )
        print( "[RPI] rfreq: " + ", ".join( [ "%.1lf"%( math.fabs( i ) ) for i in frq[0:7] ] ) + " _cm^-1" )
        print( "[RPI] l_rQV: %20.10le"%( rQV ) )
        # vibrational partition function: instanton (recycling self.hess)
        hess = numpy.zeros( ( size, size ), dtype=numpy.float64 )
        rdim = self.dime * 3
        self.coor[0:self.size] = o_coor
        j_cc = 0
        for i in range( self.half + 1 ):
            i_cc = i * rdim
            hess[j_cc:j_cc+dime,j_cc:j_cc+dime] = self.hess[i_cc:i_cc+rdim,i_cc:i_cc+rdim]
            j_cc += dime
        x_cc = self.size
        for i in range( self.half - 1, 0, -1 ):
            i_cc = i * rdim
            hess[j_cc:j_cc+dime,j_cc:j_cc+dime] = self.hess[i_cc:i_cc+rdim,i_cc:i_cc+rdim]
            j_cc += dime
            i_cc = i * self.dime
            self.coor[x_cc:x_cc+self.dime] = o_coor[i_cc:i_cc+self.dime]
            x_cc += self.dime
        for i in range( self.bead ):
            j_cc = i * dime
            if( i == 0 ):
                j_mm = size - dime
                j_pp = dime
            elif( i == self.bead - 1 ):
                j_mm = j_cc - dime
                j_pp = 0
            else:
                j_mm = j_cc - dime
                j_pp = j_cc + dime
            for j in range( dime ):
                kmb = self.mass[j//3][0] * self.kumb
                hess[j_cc+j,j_mm+j] -= 2.0 * kmb
                hess[j_cc+j,j_pp+j] -= 2.0 * kmb
        frq = qm3.utils.hessian.frequencies( self, hess )[0]
        tQV = - numpy.sum( numpy.log( 2.0 * numpy.sinh( frq[7:] * cte * 0.5 ) ) )
        tQV -= math.log( 2.0 * math.sinh( math.fabs( frq[0] ) * cte * 0.5 ) )
        print( "[RPI] tfreq: " + ", ".join( [ "%.1lf"%( math.fabs( i ) ) for i in frq[0:9] ] ) + " _cm^-1" )
        print( "[RPI] l_tQV: %20.10le"%( tQV ) )
        self.actv = o_actv
        self.mass = o_mass
        self.coor = o_coor
        # beads exchange
        tQP = 0.0
        for i in range( self.half + 1 ):
            i_cc = i * self.dime
            if( i == 0 ):
                scal = 1.0
                i_mm = ( i + 1 ) * self.dime
            elif( i == self.half ):
                scal = 1.0
                i_mm = ( i - 1 ) * self.dime
            else:
                scal = 2.0
                i_mm = ( i - 1 ) * self.dime
            for j in range( self.dime ):
                tQP += scal * self.mass[j] * numpy.sum( numpy.square( self.coor[i_cc+j] - self.coor[i_mm+j] ) )
        cte = 2.0 * math.pi * self.bead * self.temp * qm3.data.KB * 1.0e-23 / ( qm3.data.NA * qm3.data.H * qm3.data.H )
        tQP = 0.5 * math.log( cte * tQP )
        print( "[RPI] l_tQP: %20.10le"%( tQP ) )
        # kinetic constant
        self.k_RPI = qm3.data.KB * self.temp * self.bead / qm3.data.H * math.exp( tQP + tQV + tQR - rQV - rQR + efunc )
        print( "[RPI] k_cin: %20.10le _1/s"%( self.k_RPI  ) )
        self.kappa = self.k_RPI / self.k_TST
        print( "      kappa: %20.10le"%( self.kappa ) )
        print( "  dE_tunnel: %.2lf _kJ/mol"%( round( - 8.314 * math.log( self.kappa ) * self.temp / 1000., 2 ) ) )


    def plot( self, r_func: float, t_func: float ):
        try:
            import  matplotlib.pyplot
            matplotlib.pyplot.clf()
            matplotlib.pyplot.grid( True )
            matplotlib.pyplot.ylabel( "E [kJ/mol]" )
            matplotlib.pyplot.xlabel( "replica" )
            matplotlib.pyplot.plot( self.ener, '-o' )
            matplotlib.pyplot.tight_layout()
            matplotlib.pyplot.show()
        except:
            pass
        arc = numpy.zeros( self.dime, dtype=numpy.float64 )
        for i in range( 1, self.half + 1 ):
            i_cc = i * self.dime
            arc += numpy.sqrt( numpy.sum( numpy.square( self.coor[i_cc:i_cc+self.dime] - self.coor[i_cc-self.dime:i_cc] ), axis = 1 ) )
        print( "\n%-8s%-8s%8s"%( "Atom", "Label", "Arc [A]" ) )
        print( 24 * "-" )
        for i in range( self.dime ):
            print( "%-8d%-8s%8.3lf"%( i+1, self.mole.labl[self.sele[i]], arc[i] ) )
        print( 24 * "-" )
        with open( "rpi_arc.xyz", "wt" ) as f:
            f.write( "%d\n\n"%( self.dime ) )
            i_cc = 0
            for j in range( self.dime ):
                f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%(
                    qm3.data.symbol[self.mole.anum[self.sele[j]]],
                    self.coor[i_cc+j,0], self.coor[i_cc+j,1], self.coor[i_cc+j,2] ) )
            f.write( "%d\n\n"%( self.dime ) )
            i_cc = self.half * self.dime
            for j in range( self.dime ):
                f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%(
                    qm3.data.symbol[self.mole.anum[self.sele[j]]],
                    self.coor[i_cc+j,0], self.coor[i_cc+j,1], self.coor[i_cc+j,2] ) )
            for i in range( 1, self.half ):
                f.write( "%d\n\n"%( self.dime ) )
                i_cc = i * self.dime
                for j in range( self.dime ):
                    f.write( "%-2s%20.10lf%20.10lf%20.10lf\n"%( "X",
                        self.coor[i_cc+j,0], self.coor[i_cc+j,1], self.coor[i_cc+j,2] ) )

