#!/usr/bin/env python3
import  sys
import  numpy
import  openmm
import  openmm.app
import  openmm.unit
import	qm3
import  qm3.engines.mopac
import  qm3.utils._dcd
import  _mmdec
import  exclusions
import  pickle
import  collections


mol = qm3.molecule()
mol.pdb_read( open( sys.argv[1] ) )
mol.boxl = numpy.array( [ 92.154, 102.242, 97.285 ] )
mol.prmtop_read( open( "prmtop" ) )

_top = openmm.app.amberprmtopfile.AmberPrmtopFile( "prmtop" )
_sys = _top.createSystem(
    nonbondedMethod = openmm.app.CutoffNonPeriodic,
    nonbondedCutoff = 16.0 * openmm.unit.angstrom,
    switchDistance = 14.5 * openmm.unit.angstrom,
    implicitSolvent = None,
    rigidWater = False )

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_QM.pk", "rb" ) as f:
    iqm = pickle.load( f )
    sqm[iqm] = True
smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
with open( "sele_MM.pk", "rb" ) as f:
    imm = pickle.load( f )
    smm[imm] = True
with open( "sele_LA.pk", "rb" ) as f:
    sla = pickle.load( f )

for k in range( _sys.getNumForces() ):
    cur = _sys.getForce( k )
    if( type( cur ) == openmm.openmm.NonbondedForce ):
        mol.epsi = numpy.zeros( mol.natm, dtype=numpy.float64 )
        mol.rmin = numpy.zeros( mol.natm, dtype=numpy.float64 )
        for i in range( mol.natm ):
            q,s,e = cur.getParticleParameters( i )
            mol.rmin[i] = s.value_in_unit( openmm.unit.angstrom )
            mol.epsi[i] = e.value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
        mol.rmin *= 0.5612310241546865
        mol.epsi = numpy.sqrt( mol.epsi )
    if( type( cur ) == openmm.openmm.HarmonicBondForce ):
        bnd = []
        for i in range( cur.getNumBonds() ):
            ai, aj, r0, ku = cur.getBondParameters( i )
            bnd.append( [ ai, aj ] )

emm = _mmdec.QMLJ( mol, iqm, imm, exclusions.exclusions( mol.natm, bnd, sqm ) )

eqm = qm3.engines.mopac.run( mol, "AM1", 0, sel_QM = sqm, sel_MM = smm, link = sla )

res = collections.OrderedDict()
for i in imm:
    if( mol.resn[i] in [ "WAT" ] ):
        key = "BLK"
    elif( mol.resn[i] in [ "Na+" ] ):
        key = "CIO"
    else:
        key = "%s:%s:%d"%( mol.segn[i], mol.resn[i], mol.resi[i] )
    if( not key in [ "A:CYS:145", "A:HID:41", "A:V2M:613" ] ):
        if( key in res ):
            res[key].append( i )
        else:
            res[key] = [ i ]



def deco( mol, res, eqm, emm, fds ):
    mol.func = 0.0
    eqm.get_func( mol, 1000 )
    print( "    full_QM: %20.10lf"%( mol.func ) )
    chg = mol.chrg.copy()
    mol.chrg = numpy.zeros( mol.natm, dtype=numpy.float64 )
    mol.func = 0.0
    eqm.get_func( mol, -1 )
    vac = mol.func
    print( "     vac_QM: %20.10lf"%( mol.func ) )
    mol.evdw = numpy.zeros( mol.natm, dtype=numpy.float64 )
    emm.get_func( mol )
    print( "    full_LJ: %20.10lf"%( mol.evdw.sum() ) )
    _qm = []
    _mm = []
    for k in res.keys():
        mol.chrg = numpy.zeros( mol.natm, dtype=numpy.float64 )
        for j in res[k]:
            mol.chrg[j] = chg[j]
        mol.func = 0
        eqm.get_func( mol, -1 )
        _qm.append( mol.func )
        _mm.append( sum( [ mol.evdw[j] for j in res[k] ] ) )
        print( "             %20.10lf%20.10lf  %s"%( mol.func - vac, _mm[-1], k ) )
    mol.chrg = chg[:]
    fds[0].write( "%20.10lf\n"%( vac ) )
    fds[0].flush()
    fds[1].write( "".join( [ "%20.10lf"%( i ) for i in _qm ] ) + "\n" )
    fds[1].flush()
    fds[2].write( "".join( [ "%20.10lf"%( i ) for i in _mm ] ) + "\n" )
    fds[2].flush()



fds = [ open( w, "wt" ) for w in [ "__.vac", "__.ele", "__.vdw" ] ]
deco( mol, res, eqm, emm, fds )
if( len( sys.argv ) == 3 ):
    dcd = qm3.utils._dcd.dcd()
    dcd.open_read( sys.argv[2] )
    i = 0
    while( dcd.next( mol ) and i < trm ):
        print( "-- frame: ", i )
        deco( mol, res, eqm, emm, fds )
        i += 1
    dcd.close()
for w in fds:
    w.close()
