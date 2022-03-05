import  sys
import	numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3.mol
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.sander
import  qm3.actions.minimize
import  qm3.problem


class my_problem( qm3.problem.template ):
    def __init__( self ):
        qm3.problem.template.__init__( self )

        self.mol = qm3.mol.molecule( "../amber.pdb" )
        self.mol.boxl = [ 25.965, 29.928, 28.080 ]
        qm3.engines.sander.topology_read( self.mol, "../amber.prmtop" )

        _top = openmm.app.amberprmtopfile.AmberPrmtopFile( "../amber.prmtop" )
        _sys = _top.createSystem(
            nonbondedMethod = openmm.app.CutoffPeriodic,
            nonbondedCutoff = 12.0 * openmm.unit.angstrom,
            switchDistance = 10.0 * openmm.unit.angstrom,
            implicitSolvent = None,
            rigidWater = False )
        _sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3( self.mol.boxl[0], 0.0, 0.0 ) * openmm.unit.angstrom,
            openmm.Vec3( 0.0, self.mol.boxl[1], 0.0 ) * openmm.unit.angstrom,
            openmm.Vec3( 0.0, 0.0, self.mol.boxl[2] ) * openmm.unit.angstrom )

        sqm = [ i for i in range( self.mol.natm ) if self.mol.resn[i] == "SUS" ]
        smm = self.mol.sph_sel( sqm, 10 )
        print( len( sqm ), len( smm ) )

        self.emm = qm3.engines.openmm.run_native( _sys, _top, qm_atom = sqm, platform = "OpenCL" )
        self.eqm = qm3.engines.xtb.run_dynlib( self.mol, 1, 0, sqm, smm )

        for i in sqm:
            self.mol.chrg[i] = 0.0
        self.emm.update_chrg( self.mol )

        self.size = 3 * self.mol.natm
        self.coor = self.mol.coor


    def get_func( self ):
        self.mol.func = 0.0
        self.emm.get_func( self.mol )
        self.eqm.get_func( self.mol )
        self.func = self.mol.func


    def get_grad( self ):
        self.mol.func = 0.0
        self.mol.grad = [ 0.0 for i in range( self.size ) ]
        self.emm.get_grad( self.mol )
        self.eqm.get_grad( self.mol )
        self.func = self.mol.func
        self.grad = self.mol.grad


obj = my_problem()
qm3.actions.minimize.fire( obj, step_number = 1000, print_frequency = 100 )
obj.get_grad()
print( obj.func )
print( numpy.linalg.norm( obj.grad ) )
