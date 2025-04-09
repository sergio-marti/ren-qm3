#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import  qm3
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.utils
import  qm3.utils.hessian
import  qm3.actions.minimize
import  sys
import  os

mol = qm3.molecule()
mol.pdb_read( open( "node.25" ) )
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()

_psf = openmm.app.charmmpsffile.CharmmPsfFile( "oxy-cope.psf" )
_psf.setBox( mol.boxl[0] * openmm.unit.angstrom,
        mol.boxl[1] * openmm.unit.angstrom,
        mol.boxl[2] * openmm.unit.angstrom )
_prm = openmm.app.charmmparameterset.CharmmParameterSet( "oxy-cope.top", "oxy-cope.prm" )
_sys = _psf.createSystem( _prm,
    nonbondedMethod = openmm.app.CutoffPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.0 * openmm.unit.angstrom,
    rigidWater = False )

sqm = mol.resn == "COP"
smm = mol.sph_sel( sqm, 14 )
print( sqm.sum(), smm.sum(), end = " " )
smm = numpy.logical_and( smm, numpy.logical_not( sqm ) )
print( smm.sum() )

emm = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )
eqm = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )

mol.engines["mm"] = emm
mol.engines["qm"] = eqm

mol.set_active( sqm )

log = open( "borra_log.mm", "wt" )


def calc_hess( self: object, step: int ):
    eqm.get_func( self )
    self.set_active( smm )
    self.engines["mm"].update_chrg( self )
    self.engines.pop( "qm" )
    qm3.actions.minimize.fire( self, gradient_tolerance = 0.5, log_file = log )
    log.flush()
    self.chrg[sqm] = 0.0
    self.engines["mm"].update_chrg( self )
    self.engines["qm"] = eqm
    self.set_active( sqm )
    if( step % 10 == 0 ):
        self.hess = qm3.utils.hessian.numerical( self )
        qm3.utils.hessian.manage( self, self.hess )
        self.get_grad()
    else:
        self.get_grad()
        qm3.utils.hessian.manage( self, self.hess, should_update = True )
    return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) )


qm3.actions.minimize.baker( mol,
        calc_hess,
        gradient_tolerance = 2.0,
        step_number = 100,
        print_frequency = 1,
        follow_mode = 0 )

with open( "saddle.pdb", "wt" ) as f:
    mol.pdb_write( f )

val, vec = qm3.utils.hessian.frequencies( mol, mol.hess )
print( val[0:10] )
qm3.utils.hessian.normal_mode( mol, val, vec, 0, afac = 8.0 )
