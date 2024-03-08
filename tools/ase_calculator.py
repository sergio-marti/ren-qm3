#!/usr/bin/env python3
import  sys
import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm

sys.path.insert( 0, "/Users/smarti/Devel/ase/ase" )
import  ase.calculators.calculator
import  ase.io
import  ase.optimize
import  ase.units
import  ase.md
import  ase.md.velocitydistribution


class ase_calculator( ase.calculators.calculator.Calculator ):
    implemented_properties = [ "energy", "forces" ]
    nolabel = True
    def __init__( self, **kwargs ):
        ase.calculators.calculator.Calculator.__init__( self, **kwargs )

        box  = [ 26.7526393, 29.0550478, 24.9421203 ]
        self.mol = qm3.molecule()
        self.mol.prmtop_read( open( "adp.parm7", "rt" ) )
        self.mol.pdb_read( open( "adp.pdb", "rt" ) )
        self.mol.boxl = numpy.array( box )
        _top = openmm.app.amberprmtopfile.AmberPrmtopFile( "adp.parm7" )
        _sys = _top.createSystem(
            nonbondedMethod = openmm.app.CutoffPeriodic,
            nonbondedCutoff = 12.0 * openmm.unit.angstrom,
            rigidWater = False,
            implicitSolvent = None,
            switchDistance = 10.0 * openmm.unit.angstrom )
        _sys.setDefaultPeriodicBoxVectors(
            openmm.Vec3( box[0], 0.0, 0.0 ) * openmm.unit.angstrom,
            openmm.Vec3( 0.0, box[1], 0.0 ) * openmm.unit.angstrom,
            openmm.Vec3( 0.0, 0.0, box[2] ) * openmm.unit.angstrom )
        self.mol.engines["mm"] = qm3.engines.openmm.run( _sys, _top, platform = "CPU" )
        self.mol.set_active()

    def calculate( self, atoms = None, properties = None, system_changes = ase.calculators.calculator.all_changes ):
        if( properties is None ):
            properties = self.implemented_properties
        ase.calculators.calculator.Calculator.calculate( self, atoms, properties, system_changes )
        # default length: Angstroms
        self.mol.coor = atoms.get_positions()
        self.mol.get_grad()
        # default energy: kJ/mol >> eV
        self.results[ "energy" ] = self.mol.func * 0.010364269574711572
        self.results[ "forces" ] = - self.mol.grad * 0.010364269574711572



calc = ase_calculator()

atm = ase.io.read( "adp.pdb" )
atm.set_cell( calc.mol.boxl )
atm.calc = calc

opt = ase.optimize.FIRE( atm, trajectory = "ase_calculator.opt.traj", logfile = "ase_calculator.log" )
opt.run( fmax = 0.5 )

ase.md.velocitydistribution.MaxwellBoltzmannDistribution( atm, temperature_K = 300 )
dyn = ase.md.langevin.Langevin( atm, timestep = 1.0 * ase.units.fs, temperature_K = 300,
            friction = 0.02, trajectory = "ase_calculator.dyn.traj", logfile = "ase_calculator.log" )
dyn.run( 1000 )
