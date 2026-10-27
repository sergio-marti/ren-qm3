import  os
os.environ["OMP_NUM_THREADS"] = "1"
import	numpy
import	qm3
import  qm3.engines
import  io
import  sys

#
# link atoms: analytic gradients vs central finite differences on the QM host (C10), the MM partner (C6),
# a QM atom and the closest MM atoms (CHARMM sample, link C10-C6)
#
#   python3 test_link_chain.py toy           analytic model energy of the QM + link-H coordinates (no QM program)
#   python3 test_link_chain.py pyscf         qm3.engines.pyscf.run (CPU)
#   python3 test_link_chain.py orca [exe]    qm3.engines.orca.run   (exe: command line, default "orca orca.inp > orca.out")
#

cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep
who = sys.argv[1] if len( sys.argv ) > 1 else "toy"


mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.guess_atomic_numbers()

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]


class toy( qm3.engines.template ):
    """
    E = sum_i k ( x_i - x0_i )^2 + sum_{i<j} q / r_ij over the QM atoms and the link H (no MM interaction):
    any error in the link-atom chain rule shows on C10 and C6
    """
    def __init__( self, mol, sel_QM, sel_MM, link ):
        qm3.engines.template.__init__( self, mol, sel_QM, sel_MM, link )
        self.ref = self.coor( mol ) + 0.1

    def coor( self, mol ):
        crd = [ mol.coor[i] for i in self.sel ]
        for i,j in self.lnk:
            crd.append( qm3.engines.Link_coor( i, j, mol )[0] )
        return( numpy.array( crd ) )

    def get_func( self, mol ):
        x = self.coor( mol )
        r = numpy.linalg.norm( x[:,None,:] - x[None,:,:], axis = 2 ) + numpy.eye( len( x ) )
        out = 10.0 * numpy.sum( numpy.square( x - self.ref ) ) + numpy.sum( numpy.triu( 100.0 / r, 1 ) )
        mol.func += out
        return( out )

    def get_grad( self, mol ):
        out = self.get_func( mol )
        x = self.coor( mol )
        d = x[:,None,:] - x[None,:,:]
        r = numpy.linalg.norm( d, axis = 2 ) + numpy.eye( len( x ) )
        g = 20.0 * ( x - self.ref ) - numpy.sum( ( 100.0 / r ** 3 * ( 1.0 - numpy.eye( len( x ) ) ) )[:,:,None] * d, axis = 1 )
        mol.grad[self.sel,:] += g[:len( self.sel )]
        qm3.engines.Link_chain( mol, self.lnk, g[len( self.sel ):] )
        return( out )


if( who == "toy" ):
    mol.engines["qm"] = toy( mol, sqm, smm, sla )
    dsp = 1.e-5
elif( who == "pyscf" ):
    import  qm3.engines.pyscf
    opt = { "basis": "sto-3g", "conv_tol": 1.e-12, "charge": 1, "spin": 0, "method": "lda,vwn",
        "memory": 4096, "grid": 3, "max_cyc": 100, "nproc": 4 }
    mol.engines["qm"] = qm3.engines.pyscf.run( mol, opt, sqm, smm, sla )
    dsp = 1.e-4
elif( who == "orca" ):
    import  qm3.engines.orca
    f = io.StringIO( """qm3_charges
! qm3_job rks lda sto-3g verytightscf nopop
*xyz 1 1
qm3_atoms
*
""" )
    mol.engines["qm"] = qm3.engines.orca.run( mol, f, sqm, smm, sla )
    mol.engines["qm"].exe = sys.argv[2] if len( sys.argv ) > 2 else "orca orca.inp > orca.out"
    dsp = 1.e-3

mol.get_grad()
ana = mol.grad.copy()
print( "E = %.6lf kJ/mol"%( mol.func ) )

cen = numpy.mean( mol.coor[sqm], axis = 0 )
dst = numpy.linalg.norm( mol.coor - cen, axis = 1 )
c10 = mol.indx["A"][1]["C10"]
c06 = mol.indx["A"][1]["C6"]
iqm = [ i for i in numpy.flatnonzero( sqm ) if i != c10 ][:1]
imm = [ i for i in numpy.argsort( dst ) if smm[i] and not sqm[i] and i not in [ c10, c06 ] ][:2]
err = 0.0
for i in [ c10, c06 ] + iqm + imm:
    for j in [0, 1, 2]:
        bak = mol.coor[i,j]
        mol.coor[i,j] = bak + dsp
        fp = mol.get_func()
        mol.coor[i,j] = bak - dsp
        fm = mol.get_func()
        mol.coor[i,j] = bak
        num = ( fp - fm ) / ( 2.0 * dsp )
        err = max( err, abs( num - ana[i,j] ) )
        print( "%5d %-4s %s %14.6lf %14.6lf %10.2e"%( i, mol.labl[i], "QM" if sqm[i] else "MM", ana[i,j], num, num - ana[i,j] ) )
print( "max |analytic - numerical| = %.2e kJ/(mol A)"%( err ) )
