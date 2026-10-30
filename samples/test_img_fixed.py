import  os
os.environ["OMP_NUM_THREADS"] = "1"
import	numpy
import	qm3
import  qm3.data
import  qm3.engines
import  sys

#
# image of the MM charges in a non-periodic embedding, along a path x(s) = x0 + s * dx
#
#   python3 test_img_fixed.py
#
#   img = True re-chooses the minimum image of every MM charge at every call: charges near half a box
#   from the QM centroid jump to the other side, E(s) is discontinuous and E(s) - E(0) drifts away from
#   the work integral of g.dx. At a fixed geometry energy and gradient are consistent, so single-point
#   finite differences never see it: only motion does (MD, or this scan).
#   img = "fixed" keeps the image of the first call and the embedding is conservative.
#

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


class coulomb( qm3.engines.template ):
    """fixed QM charges + MM point charges: the cheapest engine with the same embedding as the QM codes"""
    def __init__( self, mol, sel_QM, sel_MM ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM )

    def get_grad( self, mol ):
        crd = self.mm_coor( mol )
        qmc = mol.coor[self.sel]
        dr  = qmc[:,None,:] - crd[None,:,:]
        r   = numpy.sqrt( ( dr * dr ).sum( axis = 2 ) )
        qq  = qm3.data.H2J * qm3.data.A0 * mol.chrg[self.sel][:,None] * mol.chrg[self.nbn][None,:]
        mol.func += numpy.sum( qq / r )
        g   = - ( qq / ( r * r * r ) )[:,:,None] * dr
        mol.grad[self.sel] += g.sum( axis = 1 )
        mol.grad[self.nbn] -= g.sum( axis = 0 )


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.guess_atomic_numbers()
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = numpy.logical_not( sqm )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]

# displacement: the MM charges close to half a box from the QM centroid cross it along the path
x0  = mol.coor.copy()
cen = numpy.mean( x0[sqm], axis = 0 )
frc = ( x0 - cen ) / mol.boxl
frc -= numpy.round( frc, 0 )
mov = numpy.logical_and( smm, numpy.max( numpy.abs( frc ), axis = 1 ) > 0.47 )
dx  = numpy.zeros( x0.shape )
dx[mov] = 0.10 * mol.boxl * numpy.sign( frc[mov] )
numpy.random.seed( 1 )
dx += 0.05 * numpy.random.normal( size = x0.shape )
print( "QM atoms: %d   MM charges: %d   moved across half a box: %d"%( sqm.sum(), smm.sum(), mov.sum() ) )


def scan( eng, ns = 40 ):
    mol.engines = { "qm": eng }
    ds  = 1.0 / ns
    ene = []
    gdx = []
    wrk = [ 0.0 ]
    for k in range( ns + 1 ):
        mol.coor = x0 + k * ds * dx
        mol.get_grad()
        ene.append( mol.func )
        gdx.append( numpy.sum( mol.grad * dx ) )
        if( k > 0 ):
            wrk.append( wrk[-1] + 0.5 * ds * ( gdx[-1] + gdx[-2] ) )
    mol.coor = x0
    dlt = numpy.array( ene ) - ene[0] - numpy.array( wrk )
    print( "img = %-7s  E(1) - E(0) = %10.3lf   work = %10.3lf   max |E - E0 - W| = %8.3lf   max step = %8.3lf kJ/mol"%(
        str( eng.img ), ene[-1] - ene[0], wrk[-1], numpy.max( numpy.abs( dlt ) ), numpy.max( numpy.abs( numpy.diff( dlt ) ) ) ) )
    return( numpy.max( numpy.abs( numpy.diff( dlt ) ) ) )


def check( make ):
    res = {}
    for img in [ True, "fixed" ]:
        eng = make()
        eng.img = img
        res[img] = scan( eng )
    print( "img = fixed conservative (max step < 0.1 kJ/mol):", res["fixed"] < 0.1 )
    print( "img = True discontinuous (max step > 1 kJ/mol):  ", res[True] > 1.0 )


print( "\n-- point charges" )
check( lambda: coulomb( mol, sqm, smm ) )

