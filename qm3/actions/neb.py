import  math
import  numpy
import  typing


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



class serial( object ):
    """
    the value of 'kumb' should be approx the same of the potential energy barrier

    when optimizing the whole band, set the 'gradient_tolerance' equal to [0.1:0.5] * nodes (_kJ/mol.A)

    J. Chem. Phys. v113, p9978 (2000) [10.1063/1.1323224]
    """
    def __init__( self, mol: object, guess: list, kumb: float ):
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
#            fum = self.kumb * numpy.sum( ( dcm - dcM ) * tau )
#            gum = fum * tau
#            return( tau, fum * numpy.sum( ( dcm - dcM ) * tau ), gum )
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
#            tau, fum, gum = __calc_tau( vpot[who-1], vpot[who], vpot[who+1],
            tau, gum = __calc_tau( vpot[who-1], vpot[who], vpot[who+1],
                    self.coor[ii-self.dime:ii],
                    self.coor[ii:jj],
                    self.coor[jj:jj+self.dime] )
#            self.func += fum * 0.5
            self.grad[ii:jj] += gum - numpy.sum( tau * self.grad[ii:jj] ) * tau



class parall( object ):
    """
    the value of 'kumb' should be approx the same of the potential energy barrier

    when optimizing the whole band, set the 'gradient_tolerance' equal to [0.1:0.5] * nodes (_kJ/mol.A)

    J. Chem. Phys. v113, p9978 (2000) [10.1063/1.1323224]
    """
    def __init__( self, mol: object, guess: list, kumb: float, chunks: list, opar: object ):
        self.mole = mol
        self.kumb = kumb
        self.sele = numpy.argwhere( mol.actv.ravel() ).ravel()
        self.dime = len( self.sele )
        self.node = len( guess )
        self.chnk = chunks[:]
        self.ncpu = len( chunks )
        self.opar = opar
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
#            fum = self.kumb * numpy.sum( ( dcm - dcM ) * tau )
#            gum = fum * tau
#            return( tau, fum * numpy.sum( ( dcm - dcM ) * tau ), gum )
            gum = self.kumb * numpy.sum( ( dcm - dcM ) * tau ) * tau
            return( tau, gum )
        # ----------------------------------------------------------------------
        vpot = [ 0.0 for i in range( self.node ) ]
        self.grad = numpy.zeros( ( self.natm, 3 ), dtype=numpy.float64 )
        # sync coordinates to nodes (chunks)
        self.opar.barrier()
        for who in range( 1, self.ncpu ):
            self.opar.send_i4( who, [ 1 ] )
            tmp = []
            for itm in self.chnk[who]:
                ii  = itm * self.dime
                tmp += self.coor[ii:ii+self.dime].ravel().tolist()
            self.opar.send_r8( who, tmp )
        # individuals (own chunk)
        for who in self.chnk[0]:
            ii = who * self.dime
            self.mole.coor[self.sele] = self.coor[ii:ii+self.dime]
            self.mole.get_grad()
            self.mole.project_gRT()
            vpot[who] = self.mole.func
            self.grad[ii:ii+self.dime] = self.mole.grad[self.sele]
            self.neb_data( who )
        # sync function and gradients from nodes (chunks)
        self.opar.barrier()
        for who in range( 1, self.ncpu ):
            siz = len( self.chnk[who] )
            fun = self.opar.recv_r8( who, siz )
            tmp = siz * self.dime * 3
            grd = numpy.array( self.opar.recv_r8( who, tmp ) )
            grd.shape = ( self.dime * siz, 3 )
            for i in range( len( self.chnk[who] ) ):
                vpot[self.chnk[who][i]] = fun[i]
                ii = self.chnk[who][i] * self.dime
                jj = i * self.dime
                self.grad[ii:ii+self.dime] = grd[jj:jj+self.dime]
        self.func = sum( vpot )
        # connections (first and last are fixed references)
        for who in range( 1, self.node - 1 ):
            ii = who * self.dime
            jj = ii + self.dime
#            tau, fum, gum = __calc_tau( vpot[who-1], vpot[who], vpot[who+1],
            tau, gum = __calc_tau( vpot[who-1], vpot[who], vpot[who+1],
                    self.coor[ii-self.dime:ii],
                    self.coor[ii:jj],
                    self.coor[jj:jj+self.dime] )
#            self.func += fum * 0.5
            self.grad[ii:jj] += gum - numpy.sum( tau * self.grad[ii:jj] ) * tau
