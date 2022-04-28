import  math
import  numpy
import  typing
#import  collections
import  qm3.utils.interpolation


def distribute( rcrd: numpy.array, rmet: list,
        interpolant: object = qm3.utils.interpolation.hermite_spline ) -> numpy.array:
    nwin, ncrd = rcrd.shape
    arcl = numpy.zeros( self.nwin, dtype=numpy.float64 )
    for i in range( 1, self.nwin ):
        vec = rcrd[i] - rcrd[i-1]
        vec.shape = ( ncrd, 1 )
        mat = 0.5 * ( rmet[i] + rmet[i-1] )
        mat = numpy.linalg.inv( mat )
        arcl[i] = math.sqrt( numpy.dot( vec.T, numpy.dot( mat, vec ) ) )
    arcl = numpy.cumsum( arcl )
    delz = arcl[-1] / float( nwin - 1.0 )
    equi = numpy.array( [ i * delz for i in range( nwin ) ], dtype=numpy.float64 )
    ocrd = numpy.zeros( ( nwin, ncrd ), dtype=numpy.float64 )
    for j in range( ncrd ):
        inte = interpolant( arcl, rcrd[:,j] )
        ocrd[:,j] = numpy.array( [ inte.calc( x )[0] for x in equi ], dtype=numpy.float64 )
    return( ocrd )



class string( object ):
    """
    tstp: dt / gamma [ps^2] ~  time_step / distribution_frequency [0.001/100 = 1e-5]

    -------------------------------------
    ncrd        nwin
    atom_1,i    atom_1,j    kumb_1
    ...         ...
    atom_nc,i   atom_nc,j   kumb_nc
    iref_1,1    iref_1,nc
    ...         ...
    iref_nw,1   iref_nw,nc
    -------------------------------------

    kumb_i: kJ / ( mol Angs^2 )

    Chem. Phys. Lett. v446, p182 (2007) [10.1016/j.cplett.2007.08.017]
    J. Comput. Chem. v35, p1672 (2014) [10.1002/jcc.23673]
    J. Phys. Chem. A v121, p9764 (2017) [10.1021/acs.jpca.7b10842]
    """
    def __init__( self, mol: object, node: int, tstp: float,
            str_cnf: typing.IO ):
        self.node = node
        self.tstp = tstp
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
        # store the masses (and their square root)
        self.mass = mol.mass[list( self.jidx.keys() )]
        self.mass = numpy.column_stack( ( self.mass, self.mass, self.mass ) ).reshape( self.jcol )
        self.sqms = numpy.sqrt( self.mass ).reshape( ( self.jcol // 3, 3 ) ) 
        print( self.sqms )
        # initialize metrics
        self.cmet = numpy.zeros( ( self.ncrd, self.ncrd ), dtype=numpy.float64 )


    def get_grad( self, mol: object ) -> numpy.array:
        # calculate current CVs and Jacobian
        ccrd = numpy.zeros( self.ncrd )
        jaco = numpy.zeros( ( self.ncrd, self.jcol ) )
        for i in range( self.ncrd ):
            ai = self.atom[i][0]
            aj = self.atom[i][1]
            rr = mol.coor[aj] - mol.coor[ai]
            ccrd[i] = numpy.linalg.norm( rr )
            for j in [0, 1, 2]:
                jaco[i,3*self.jidx[ai]+j] -= rr[j] / ccrd[i]
                jaco[i,3*self.jidx[aj]+j] += rr[j] / ccrd[i]
        # translate umbrella gradients into molecule
        diff = self.kumb * ( ccrd - self.rcrd )
        grad = numpy.dot( diff, jaco )
        grad.shape = ( self.jcol // 3, 3 )
        mol.grad[list( self.jidx.keys() ),:] += grad
        # calculate current metric tensor (eq. 7 @ 10.1016/j.cplett.2007.08.017)
        for i in range( self.ncrd ):
            for j in range( i, self.ncrd ):
                self.cmet[i,j] = numpy.sum( jaco[i,:] * jaco[j,:] / self.mass )
                self.cmet[j,i] = self.cmet[i,j]
        # perform dynamics on the reference CVs (eq. 17 @ 10.1016/j.cplett.2007.08.017)
        grad = numpy.dot( diff, self.cmet )
        self.rcrd += grad * self.tstp * 100.0
