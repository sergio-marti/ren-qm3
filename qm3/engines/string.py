import  math
import  numpy
import  typing
import  qm3.utils.interpolation


def distribute( rcrd: numpy.array, rmet: list,
        interpolant: object = qm3.utils.interpolation.cubic_spline ) -> numpy.array:
    nwin, ncrd = rcrd.shape
    ccrd = rcrd.copy()
    #imet = [ None ]
    #for i in range( 1, nwin ):
    #    imet.append( numpy.linalg.inv( 0.5 * ( rmet[i] + rmet[i-1] ) ) )
    imet = [ numpy.linalg.inv( rmet[i] ) for i in range( nwin ) ]
    for ite in range( 10 ):
        arcl = numpy.zeros( nwin, dtype=numpy.float64 )
        for i in range( 1, nwin ):
            vec = ccrd[i] - ccrd[i-1]
            vec.shape = ( ncrd, 1 )
            arcl[i] = math.sqrt( numpy.dot( vec.T, numpy.dot( imet[i], vec ) )[0,0] )
        flg  = numpy.std( arcl[1:] ) / numpy.mean( arcl[1:] ) < 0.02
        arcl = numpy.cumsum( arcl )
        delz = arcl[-1] / float( nwin - 1.0 )
        equi = numpy.array( [ i * delz for i in range( nwin ) ], dtype=numpy.float64 )
        ocrd = numpy.zeros( ( nwin, ncrd ), dtype=numpy.float64 )
        for j in range( ncrd ):
            inte = interpolant( arcl, ccrd[:,j] )
            ocrd[:,j] = numpy.array( [ inte.calc( x )[0] for x in equi ], dtype=numpy.float64 )
        ccrd = ocrd.copy()
        if( flg ):
            break
    return( ocrd )


def integrate_mfep( ncrd: int, nwin: int, nstp: int,
                    cvs_lst: list, met_lst: list, frc_lst: list,
                    skip: typing.Optional[int] = 1000 ) -> numpy.array:
    # ----------------------------------------------------
    crd = numpy.zeros( ( nwin, nstp, ncrd ) )
    for i in range( nwin ):
        crd[i] = numpy.loadtxt( cvs_lst[i] )
    rcrd = numpy.mean( crd[:,skip:,:], axis = 1 )
    numpy.savetxt( "string.cvar", rcrd )
    # ----------------------------------------------------
    met = numpy.zeros( ( nwin, nstp, ncrd * ncrd ) )
    for i in range( nwin ):
        met[i] = numpy.loadtxt( met_lst[i] )
    # ----------------------------------------------------
    dFdz = numpy.zeros( ( nwin, ncrd ) )
    for i in range( nwin ):
        dFdz[i] = numpy.mean( numpy.loadtxt( frc_lst[i] )[skip:,:], axis = 0 )
    numpy.savetxt( "string.dFdz", dFdz )
    # ----------------------------------------------------
    # finite differences
    ds   = numpy.zeros( nwin )
    dzds = numpy.zeros( ( nwin, ncrd ) )
    nacc = 0.0
    for i in range( skip, nstp ):
        arcl = numpy.zeros( nwin )
        for j in range( 1, nwin ):
            dif = crd[j,i] - crd[j-1,i]
            dif.shape = ( ncrd, 1 )
            arcl[j] = numpy.sqrt( numpy.dot( dif.T, numpy.dot( numpy.linalg.inv( met[j,i].reshape( ( ncrd, ncrd ) ) ), dif ) )[0,0] )
        nacc += 1.0
        ds   += arcl
        dzds[0]  += ( crd[1,i] - crd[0,i] ) / arcl[1]
        dzds[-1] += ( crd[-1,i] - crd[-2,i] ) / arcl[-1]
        for j in range( 1, nwin - 1 ):
            dzds[j] += ( crd[j+1,i] - crd[j-1,i] ) / ( arcl[j+1] + arcl[j] )
    ds   /= nacc
    dzds /= nacc
    numpy.savetxt( "string.ds", ds )
    numpy.savetxt( "string.dzds", dzds )
    # ----------------------------------------------------
    # (eq. 20 @ 10.1016/j.cplett.2007.08.017)
    mfep = numpy.zeros( nwin )
    for i in range( 1, nwin ):
        mfep[i] = mfep[i-1] + numpy.sum( dzds[i,:] * dFdz[i,:] ) * ds[i]
    numpy.savetxt( "string.mfep", mfep )
    # ----------------------------------------------------
    try:
        import  matplotlib.pyplot as plt
        import  matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages( "string.pdf" )
        plt.clf()
        plt.grid( True )
        plt.ylabel( "distance [Å]" )
        plt.xlabel( "window" )
        for i in range( rcrd.shape[1] ):
            plt.plot( rcrd[:,i], '-' )
        plt.tight_layout()
        pdf.savefig()
        plt.clf()
        plt.grid()
        plt.ylabel( "∆F(s) [kcal/mol]" )
        plt.xlabel( "window" )
        plt.plot( mfep / 4.184, '.-' )
        plt.tight_layout()
        pdf.savefig()
        pdf.close()
    except:
        pass
    return( mfep )



class string( object ):
    """
    integrate and distribute the string each ~ 100 MD steps

    use a step_size for integration ~ 1.e-6 / 1.e-7 (default), or skip integration with 0.0 value

    -------------------------------------
    ncrd        nwin
    atom_1,i    atom_1,j    kumb_1
    ...         ...
    atom_nc,i   atom_nc,j   kumb_nc
    iref_1,1    iref_1,nc
    ...         ...
    iref_nw,1   iref_nw,nc
    -------------------------------------

    kumb_i: kJ / ( mol Angs^2 )  ~ 3000

    Chem. Phys. Lett. v446, p182 (2007) [doi:10.1016/j.cplett.2007.08.017]
    J. Comput. Chem. v35, p1672 (2014) [doi:10.1002/jcc.23673]
    J. Phys. Chem. A v121, p9764 (2017) [doi:10.1021/acs.jpca.7b10842]
    """
    def __init__( self, mol: object, node: int, str_cnf: typing.IO ):
        self.node = node
        # parse config
        tmp = str_cnf.readline().strip().split()
        self.ncrd = int( tmp[0] )
        self.nwin = int( tmp[1] )
        self.kumb = numpy.zeros( self.ncrd, dtype=numpy.float64 )
        self.atom = []
        self.jidx = {}
        for i in range( self.ncrd ):
            tmp = str_cnf.readline().strip().split()
            self.atom.append( ( int( tmp[0] ), int( tmp[1] ) ) )
            self.jidx[int(tmp[0])] = True
            self.jidx[int(tmp[1])] = True
            self.kumb[i] = float( tmp[2] )
#        self.jidx = collections.OrderedDict( { jj: ii for ii,jj in enumerate( sorted( self.jidx ) ) } )
        self.jidx = { jj: ii for ii,jj in enumerate( sorted( self.jidx ) ) }
        self.jcol = 3 * len( self.jidx )
        # load my initial reference 
        tmp = numpy.array( [ float( i ) for i in str_cnf.read().strip().split() ], dtype=numpy.float64 )
        tmp.shape = ( self.nwin, self.ncrd )
        self.rcrd = tmp[self.node,:]
        #@# store the masses (and their square root)
        #@self.mass = mol.mass[list( self.jidx.keys() )]
        #@self.mass = numpy.column_stack( ( self.mass, self.mass, self.mass ) ).reshape( self.jcol )
        # initialize variables
        self.ccrd = numpy.zeros( self.ncrd, dtype=numpy.float64 )
        self.cdif = numpy.zeros( self.ncrd, dtype=numpy.float64 )
        self.initialize_averages()


    def get_grad( self, mol: object ) -> numpy.array:
        # calculate current CVs and Jacobian
        self.ccrd = numpy.zeros( self.ncrd )
        jaco = numpy.zeros( ( self.ncrd, self.jcol ) )
        for i in range( self.ncrd ):
            ai = self.atom[i][0]
            aj = self.atom[i][1]
            rr = mol.coor[aj] - mol.coor[ai]
            self.ccrd[i] = numpy.linalg.norm( rr )
            for j in [0, 1, 2]:
                jaco[i,3*self.jidx[ai]+j] -= rr[j] / self.ccrd[i]
                jaco[i,3*self.jidx[aj]+j] += rr[j] / self.ccrd[i]
        # translate umbrella gradients into molecule
        self.cdif = self.kumb * ( self.ccrd - self.rcrd )
        grad = numpy.dot( self.cdif, jaco )
        grad.shape = ( self.jcol // 3, 3 )
        mol.grad[list( self.jidx.keys() ),:] += grad
        # calculate current metric tensor (eq. 7 @ 10.1016/j.cplett.2007.08.017)
        cmet = numpy.zeros( ( self.ncrd, self.ncrd ), dtype=numpy.float64 )
        for i in range( self.ncrd ):
            for j in range( i, self.ncrd ):
                cmet[i,j] = numpy.sum( jaco[i,:] * jaco[j,:] ) #@ / self.mass )
                cmet[j,i] = cmet[i,j]
        # accumulate
        self.nstp += 1.0
        self.amet += cmet
        self.adif += self.cdif


    def integrate( self, step_size: typing.Optional[float] = 1.e-7 ):
        self.amet /= self.nstp
        self.adif /= self.nstp
        # perform (damped) dynamics on the reference CVs (eq. 17 @ 10.1016/j.cplett.2007.08.017)
        self.rcrd += numpy.dot( self.adif, self.amet ) * step_size * 100.0


    def initialize_averages( self ):
        self.nstp = 0.0
        self.amet = numpy.zeros( ( self.ncrd, self.ncrd ), dtype=numpy.float64 )
        self.adif = numpy.zeros( self.ncrd, dtype=numpy.float64 )

