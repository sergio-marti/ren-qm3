import  math
import  numpy
import  typing
import  qm3.utils


def distribute( nodes: int, guess: list ) -> list:
    """
    guess (list) MUST contain at least the initial (coor_0) and end (coor_f) coordinate numpy.arrays
    """
    delt = []
    for i in range( 1, len( guess ) ):
        delt.append( numpy.linalg.norm( guess[i] - guess[i-1] ) )
    dtot = sum( delt )
    npts = [ int( round( delt[i] / dtot * ( nodes + 1 ), 0 ) ) for i in range( len( delt ) ) ]
    delt = []
    for i in range( 1, len( guess ) ):
        delt.append( ( guess[i] - guess[i-1] ) / npts[i-1] )
    npts[-1] += 1
    coor = []
    for i in range( len( guess ) - 1 ):
        for n in range( npts[i] ):
            coor.append( guess[i] + n * delt[i] )
    return( coor )



#
# J. Chem. Phys. v113, p9978 (2000) [10.1063/1.1323224]
#
class serial( object ):
    def __init__( self, mol: object, guess: list, kumb: float ):
        """
        the value of 'kumb' should be approx the same of the potential energy barrier

        when optimizing the whole band, set the 'gradient_tolerance' equal to [0.1:0.5] * nodes (_kJ/mol.A)
        """
        self.mole = mol
        self.kumb = kumb
        self.sele = numpy.argwhere( mol.actv.ravel() ).ravel()
        self.dime = len( self.sele )
        self.node = len( guess )
        self.natm = self.dime * self.node
        self.actv = numpy.ones( ( self.natm, 1 ), dtype=numpy.bool_ )
        self.coor = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        for i in range( self.node ):
            ii = i * self.dime
            self.coor[ii:ii+self.dime] = guess[i]


    def current_step( self, step: int ):
        pass


    def neb_data( self, node: int ):
        with open( "node.%02d"%( node ), "wt" ) as f:
            f.write( "REMARK func = %20.3lf\n"%( self.mole.func ) )
            self.mole.pdb_write( f )


    def get_grad( self ):
        # ----------------------------------------------------------------------
        def __calc_tau( potm, poti, potp, crdm, crdi, crdp ):
            dcM = crdp - crdi
            dcm = crdi - crdm
            dpM = max( math.fabs( potp - poti ), math.fabs( potm - poti ) )
            dpm = min( math.fabs( potp - poti ), math.fabs( potm - poti ) )
            if( potp > poti and poti > potm ):
                tau = dcM.copy()
            elif( potp < poti and poti < potm ):
                tau = dcm.copy()
            else:
                if( potp > potm ):
                    tau = dpM * dcM + dpm * dcm
                else:
                    tau = dpm * dcM + dpM * dcm
            tau /= numpy.linalg.norm( tau )
            gum = self.kumb * numpy.sum( ( dcm - dcM ) * tau ) * tau
            return( tau, gum )
        # ----------------------------------------------------------------------
        vpot = []
        self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        # individuals
        for who in range( self.node ):
            ii = who * self.dime
            self.mole.coor[self.sele] = self.coor[ii:ii+self.dime]
            self.mole.get_grad()
            self.mole.project_gRT()
            vpot.append( self.mole.func )
            self.grad[ii:ii+self.dime] = self.mole.grad[self.sele]
            self.neb_data( who )
        self.func = sum( vpot )
        # connections (first and last are fixed references)
        for who in range( 1, self.node - 1 ):
            ii = who * self.dime
            jj = ii + self.dime
            self.MIERDA = ( who == 1 )
            tau, gum = __calc_tau( vpot[who-1], vpot[who], vpot[who+1],
                    self.coor[ii-self.dime:ii],
                    self.coor[ii:jj],
                    self.coor[jj:jj+self.dime] )
            self.grad[ii:jj] += gum - numpy.sum( tau * self.grad[ii:jj] ) * tau


