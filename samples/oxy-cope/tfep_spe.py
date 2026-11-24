#!/usr/bin/env python3
import  os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENMM_CPU_THREADS"] = "1"
import  numpy
import  qm3
import  openmm
import  openmm.app
import  openmm.unit
import  qm3.engines.openmm
import  qm3.engines.xtb
import  qm3.engines.mopac
import  qm3.engines.mmres
import  qm3.utils._dcd
import  qm3.utils.tfep
import  pickle
import  io
import  tqdm

mol = qm3.molecule()
mol.boxl = numpy.array( [ 40.0, 40.0, 40.0 ] )
mol.psf_read( open( "oxy-cope.psf" ) )
mol.guess_atomic_numbers()
mol.xyz_read( open( "reac.xyz", "rt" ), replace = True )

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

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    sqm[pickle.load( f )] = True

mol.engines["mm"] = qm3.engines.openmm.run( _sys, _psf.topology, sel_QM = sqm, platform = "CPU" )

smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    smm[pickle.load( f )] = True

e_ll = qm3.engines.xtb.run( mol, 0, 0, sel_QM = sqm, sel_MM = smm )
e_ll.img_init( mol )

e_hl = qm3.engines.mopac.run( mol, "AM1", 0, 1, sqm, smm )
e_hl.img_init( mol )

with open( "pmf_s.delz", "rt" ) as f:
    dz = float( f.read().strip() )
cvs = qm3.engines.mmres.colvar_s( mol, "pmf_s.cnf", "s3/last.str", dz, True )

coor = []
f_ll = []
g_ll = []
f_hl = []
g_hl = []
svar = []
dvar = []
widx = []
fidx = []

idx = numpy.flatnonzero( sqm )
dcd = qm3.utils._dcd.dcd()
# ----------------------------------------------
#with tqdm.tqdm( total = 42 * 500 ) as p:
#    for w in range( 42 ):
# ----------------------------------------------
#with tqdm.tqdm( total = 5 * 500 ) as p:
#    for w in range( 5 ):
#with tqdm.tqdm( total = 7 * 500 ) as p:
#    for w in range( 15, 21 + 1 ):
with tqdm.tqdm( total = 5 * 500 ) as p:
    for w in range( 37, 42 ):
# ----------------------------------------------
        dcd.open_read( f"s3/dcd.{w:03d}" )
        k = 0
        while( dcd.next( mol.coor ) ):
            coor.append( mol.coor[idx].copy() )
            # ----------------------------------
            mol.get_grad()
            e_ll.get_grad( mol )
            f_ll.append( mol.func )
            g_ll.append( mol.grad[idx].copy() )
            # ----------------------------------
            mol.get_grad()
            e_hl.get_grad( mol )
            f_hl.append( mol.func )
            g_hl.append( mol.grad[idx].copy() )
            # ----------------------------------
            cval, ccrd = cvs.value( mol )
            svar.append( cval )
            dvar.append( ccrd )
            # ----------------------------------
            widx.append( w )
            fidx.append( k )
            k += 1
            p.update( 1 )
        dcd.close()

dat = qm3.utils.tfep.dataset( 
        idx  = idx,
        cvi  = numpy.array( sorted( cvs.indx ) ),
        crd  = numpy.array( coor ),
        s    = numpy.array( svar ),
        d    = numpy.array( dvar ),
        win  = numpy.array( widx ),
        step = numpy.array( fidx ),
        e_ll = numpy.array( f_ll ),
        g_ll = numpy.array( g_ll ),
        e_hl = numpy.array( f_hl ),
        g_hl = numpy.array( g_hl ) )
dat.save( "data.npz" )
