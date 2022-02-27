import  numpy
import  typing
import  ctypes
import  os
import  re
import  glob
import  inspect
import  qm3.data


# =================================================================================================

try:
    import  openmm
    import  openmm.app
    import  openmm.unit
    
    class qm3_openmm( object ):
        def __init__( self, omm_sys: object, omm_top: object,
                sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
                platform = "CPU" ):
            nqm = sel_QM.sum()
            if( nqm > 0 ):
                idx = numpy.argwhere( sel_QM ).ravel()
                msk = sel_QM * 1
                for ii in range( omm_sys.getNumForces() ):
                    cur = omm_sys.getForce( ii )
                    if( type( cur ) == openmm.NonbondedForce ):
                        for i in range( 0, nqm - 1 ):
                            for j in range( i + 1, nqm ):
                                cur.addException( idx[i], idx[j], 0.0, 0.0, 0.0, replace = True )
                        self.nbn = cur
                    elif( type( cur ) == openmm.HarmonicBondForce ):
                        for i in range( cur.getNumBonds() ):
                            tmp = cur.getBondParameters( i )
                            if( msk[tmp[0]] == 1 and msk[tmp[1]] == 1 ):
                                cur.setBondParameters( i, tmp[0], tmp[1], 0.0, 0.0 )
                    elif( type( cur ) == openmm.HarmonicAngleForce ):
                        for i in range( cur.getNumAngles() ):
                            tmp = cur.getAngleParameters( i )
                            if( msk[tmp[0]] + msk[tmp[1]] + msk[tmp[2]] >= 2 ):
                                cur.setAngleParameters( i, tmp[0], tmp[1], tmp[2], 0.0, 0.0 )
                    elif( type( cur ) == openmm.PeriodicTorsionForce ):
                        for i in range( cur.getNumTorsions() ):
                            tmp = cur.getTorsionParameters( i )
                            if( msk[tmp[0]] + msk[tmp[1]] + msk[tmp[2]] + msk[tmp[3]] >= 3 ):
                                cur.setTorsionParameters( i, tmp[0], tmp[1], tmp[2], tmp[3], 1, 0.0, 0.0 )
                    # charmm (improper)
                    elif( type( cur ) == openmm.CustomTorsionForce ):
                        for i in range( cur.getNumTorsions() ):
                            tmp = cur.getTorsionParameters( i )
                            if( msk[tmp[0]] + msk[tmp[1]] + msk[tmp[2]] + msk[tmp[3]] >= 3 ):
                                cur.setTorsionParameters( i, tmp[0], tmp[1], tmp[2], tmp[3], openmm.vectord( [ 0.0, tmp[4][1] ] ) )
                    # charmm (cmap)
                    elif( type( cur ) == openmm.CMAPTorsionForce ):
                        if( cur.getNumTorsions() > 0 ):
                            print( ">> there are charmm-cmaps defined...(and unhandled!)" )
    #                    for i in range( cur.getNumTorsions() ):
    #                        tmp = cur.getTorsionParameters( i )
    #                        if( msk[tmp[0]] + msk[tmp[1]] + msk[tmp[2]] + msk[tmp[3]] >= 3 ):
    #                            cur.setTorsionParameters( i, tmp[0], tmp[1], tmp[2], tmp[3], openmm.vectord( [ 0.0, tmp[4][1] ] ) )
                    elif( type( cur ) == openmm.CMMotionRemover ):
                        pass
                    else:
                        print( ">> Unhandled QM atoms at: %s [%d]"%( type( cur ), ii ) )

            self.nbn = None 
            for i in range( omm_sys.getNumForces() ):
                if( type( omm_sys.getForce( i ) ) == openmm.NonbondedForce ):
                    self.nbn = omm_sys.getForce( i )

            self.sim = openmm.app.Simulation( omm_top, omm_sys,
                openmm.CustomIntegrator( 0.001 ),
                openmm.Platform.getPlatformByName( platform ) )
    
    
        def update_chrg( self, mol: object ):
            if( self.nbn != None ):
                for i in range( mol.natm ):
                    t = self.nbn.getParticleParameters( i )
                    self.nbn.setParticleParameters( i, mol.chrg[i], t[1], t[2] )
                self.nbn.updateParametersInContext( self.sim.context )
            else:
                print( ">> Unable to update charges: no openmm.NonbondedForce available!" )
    
    
        def update_coor( self, mol: object ):
            tmp = []
            for i in range( mol.natm ):
                tmp.append( openmm.Vec3( mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] ) * openmm.unit.angstrom )
            self.sim.context.setPositions( tmp )
    
    
        def get_func( self, mol: object ):
            self.update_coor( mol )
            stt = self.sim.context.getState( getEnergy = True, getForces = False )
            mol.func += stt.getPotentialEnergy().value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
    
    
        def get_grad( self, mol: object ):
            self.update_coor( mol )
            stt = self.sim.context.getState( getEnergy = True, getForces = True )
            mol.func += stt.getPotentialEnergy().value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
            frc = stt.getForces()
            for i in range( mol.natm ):
                for j in [0, 1, 2]:
                    mol.grad[i,j] -= frc[i][j].value_in_unit( openmm.unit.kilojoule/(openmm.unit.angstrom*openmm.unit.mole) )

except:
    print( ">> OpenMM not available" )

# =================================================================================================

def Link_coor( qm_i, mm_j, mol, dst = 1.1 ):
    vv = mol.coor[mm_j] - mol.coor[qm_i]
    vv /= numpy.sqrt( numpy.dot( vv, vv ) )
    vv *= dst
    return( mol.coor[qm_i] + vv, - vv )


def Link_grad( lnk, grd ):
    for qm_i,mm_j,vec in lnk:
        m = numpy.sqrt( sum( [ k*k for k in vec ] ) )
        t = sum( [ vec[k] * ( grd[3*qm_i+k] - grd[3*mm_j+k] ) for k in [0, 1, 2] ] ) * 0.5 / m
        grd[3*qm_i:3*qm_i+3] = [ grd[3*qm_i+k] - t * vec[k] / m for k in [0, 1, 2] ]

# =================================================================================================

class qm3_mopac( object ):
    def __init__( self, mol: object, meth: str, chrg: int,
            mult: typing.Optional[int] = 1,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [],
            con: typing.Optional[float] = -1,
            cof: typing.Optional[float] = -1 ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
        hami = { "MNDO": 0, "AM1": 1, "RM1": 2, "PM3": 3, "PDDG": 4 }
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 1 + 4 * self.nQM + 4 * self.nMM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_mopac.so" )
        self.lib.qm3_mopac_setup_.argtypes = [ ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_double ),
            ctypes.POINTER( ctypes.c_double ), ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_mopac_setup_.restype = None
        self.lib.qm3_mopac_calc_.argtypes = [ ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ), ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_mopac_calc_.restype = None
        l = 1
        for i in self.sel:
            self.vec[l] = mol.anum[i]
            l += 1
        for i in range( len( self.lnk ) ):
            self.vec[l] = 1
            l += 1
        self.lib.qm3_mopac_setup_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ),
                ctypes.c_int( hami[meth] ),
                ctypes.c_int( chrg ),
                ctypes.c_int( mult ),
                ctypes.c_int( self.siz ), self.vec,
                ctypes.c_double( con ), ctypes.c_double( cof ) )


    def update_coor( self, mol: object ):
        l = 1
        for i in self.sel:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j]
                l += 1
        self.vla = []
        k = len( self.sel )
        for i in range( len( self.lnk ) ):
            c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            for j in [0, 1, 2]:
                self.vec[l] = c[j]
                l += 1
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            k += 1
        l = 1 + 4 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j] - mol.boxl[j] * numpy.round( mol.coor[i,j] / mol.boxl[j], 0 )
                l += 1
        l  = 1 + 4 * self.nQM
        for i in self.nbn:
            self.vec[l] = mol.chrg[i]
            l += 1


    def get_func( self, mol: object, maxit: typing.Optional[int] = 200 ):
        self.update_coor( mol )
        self.lib.qm3_mopac_calc_( ctypes.c_int( maxit ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1 + 3 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1


    def get_grad( self, mol: object, maxit: typing.Optional[int] = 200 ):
        self.update_coor( mol )
        self.lib.qm3_mopac_calc_( ctypes.c_int( maxit ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 1 + 3 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        g = [ self.vec[j] for j in range( 1, 3 * self.nQM + 1 ) ]
        Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        l = 1 + 4 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1

# =================================================================================================

class qm3_xtb( object ):
    def __init__( self, mol: object, chrg: int,
            nope: typing.Optional[int] = 0,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 3 + 5 * self.nQM + 4 * self.nMM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_xtb.so" )
        self.lib.qm3_xtb_calc_.argtypes = [ 
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_int ),
            ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_xtb_calc_.restype = None
        self.vec[1] = chrg
        self.vec[2] = nope
        l = 3
        for i in self.sel:
            self.vec[l] = mol.anum[i]
            l += 1
        for i in range( len( self.lnk ) ):
            self.vec[l] = 1
            l += 1
        l  = 3 + 5 * self.nQM
        for i in self.nbn:
            self.vec[l] = mol.chrg[i]
            l += 1


    def update_coor( self, mol ):
        l = 3 + self.nQM
        for i in self.sel:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j]
                l += 1
        self.vla = []
        k = len( self.sel )
        for i in range( len( self.lnk ) ):
            c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            for j in [0, 1, 2]:
                self.vec[l] = c[j]
                l += 1
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            k += 1
        l = 3 + 5 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j] - mol.boxl[j] * numpy.round( mol.coor[i,j] / mol.boxl[j], 0 )
                l += 1


    def get_func( self, mol, density = False ):
        self.update_coor( mol )
        self.lib.qm3_xtb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 3 + 4 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_xtb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 3 + 4 * self.nQM
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        l = 3 + self.nQM
        g = [ self.vec[l+j] for j in range( 3 * self.nQM ) ]
        Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        l = 3 + 5 * self.nQM + self.nMM
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1

# =================================================================================================
# sqm

# =================================================================================================

class qm3_dftb( object ):
    def __init__( self, mol: object, fdesc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
        self.tbl = { i:None for i in qm3.data.symbol[mol.anum[self.sel]] }
        if( len( self.lnk ) > 0 ):
            self.tbl["H"] = None
        self.tbl = list( self.tbl )
        self.nQM = len( self.sel ) + len( self.lnk )
        self.nMM = len( self.nbn )
        self.siz = 1 + 3 * ( self.nQM + self.nMM ) + self.nMM + self.nQM
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_dftb.so" )
        self.lib.qm3_dftb_calc_.argtypes = [
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_dftb_calc_.restype = None
        self.inp = fdesc.read()
        self.mk_input( mol, "grad" )
        self.lib.qm3_dftb_init_()


    def mk_input( self, mol, run ):
        s_qm = "  %d C\n  %s\n"%( len( self.sel ) + len( self.lnk ), str.join( " ", self.tbl ) )
        j = 0
        for i in self.sel:
            s_qm += "  %4d%4d%20.10lf%20.10lf%20.10lf\n"%( j + 1,
                    self.tbl.index( qm3.data.symbol[mol.anum[i]] ) + 1,
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
            j += 1
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            w = self.tbl.index( "H" ) + 1
            for i in range( len( self.lnk ) ):
                c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                s_qm += "  %4d%4d%20.10lf%20.10lf%20.10lf\n"%( k + 1, w, c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        s_wf = ""
        if( os.access( "charges.bin", os.R_OK ) ):
            s_wf = "  ReadInitialCharges = Yes"
        s_rn = "  CalculateForces = No"
        if( run == "grad" ):
            s_rn = "  CalculateForces = Yes"
        s_nq = ""
        if( len( self.nbn ) > 0 ):
            s_nq = str( len( self.nbn ) )
            g = open( "charges.dat", "wt" )
            for i in self.nbn:
                tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
                g.write( "%20.10lf%20.10lf%20.10lf%12.6lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] ) )
            g.close()
        f = open( "dftb_in.hsd", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_guess", s_wf )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_nchg", s_nq )
        f.write( buf )
        f.close()


    def update_coor( self, mol ):
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j]
                l += 1
        self.vla = []
        k = len( self.sel )
        for i in range( len( self.lnk ) ):
            c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            for j in [0, 1, 2]:
                self.vec[l] = c[j]
                l += 1
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            k += 1
        k = 3 * ( self.nQM + self.nMM )
        for i in self.nbn:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j] - mol.boxl[j] * numpy.round( mol.coor[i,j] / mol.boxl[j], 0 )
                l += 1
            self.vec[k] = mol.chrg[i]
            k += 1


    def get_func( self, mol, density = False ):
        self.update_coor( mol )
        self.lib.qm3_dftb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 0
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_dftb_calc_( ctypes.c_int( self.nQM ), ctypes.c_int( self.nMM ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        l = 0
        for i in self.sel:
            mol.chrg[i] = self.vec[l]
            l += 1
        g = [ j for j in self.vec[self.nQM+1:] ]
        Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1
        for i in self.nbn:
            for j in [0, 1, 2]:
                mol.grad[i,j] += self.vec[l]
                l += 1


# =================================================================================================
# dftd4

class qm3_dftd4( object ):
    def __init__( self, mol: object, parm: dict,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        self.lnk = link[:]
        self.vla = []
        self.nat = len( self.sel )
        self.all = self.nat + len( self.lnk )
        self.siz = max( 6, self.all * 3 + 1 )
        self.vec = ( ctypes.c_double * self.siz )()
        cwd = os.path.abspath( os.path.dirname( inspect.getfile( self.__class__ ) ) ) + os.sep
        self.lib = ctypes.CDLL( cwd + "_dftd4.so" )
        self.lib.qm3_dftd4_init_.argtypes = [
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_dftd4_init_.restype = None
        self.lib.qm3_dftd4_calc_.argtypes = [
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_int ),
                ctypes.POINTER( ctypes.c_double ) ]
        self.lib.qm3_dftd4_calc_.restype = None
        self.vec[0] = parm["chrg"]
        k = 1
        for i in self.sel:
            self.vec[k] = mol.anum[i]
            k += 1
        if( len( self.lnk ) > 0 ):
            for i in range( len( self.lnk ) ):
                self.vec[k] = 1
                k += 1
        self.vec[self.all+1] = parm["s6"]
        self.vec[self.all+2] = parm["s8"]
        self.vec[self.all+3] = parm["a1"]
        self.vec[self.all+4] = parm["a2"]
        self.lib.qm3_dftd4_init_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )


    def update_coor( self, mol ):
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                self.vec[l] = mol.coor[i,j]
                l += 1
        self.vla = []
        k = len( self.sel )
        for i in range( len( self.lnk ) ):
            c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
            for j in [0, 1, 2]:
                self.vec[l] = c[j]
                l += 1
            self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
            k += 1


    def get_func( self, mol, density = False ):
        self.update_coor( mol )
        self.lib.qm3_dftd4_calc_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]


    def get_grad( self, mol ):
        self.update_coor( mol )
        self.lib.qm3_dftd4_calc_( ctypes.c_int( self.all ), ctypes.c_int( self.siz ), self.vec )
        mol.func += self.vec[0]
        g = [ j for j in self.vec[1:] ]
        Link_grad( self.vla, g )
        l = 0
        for i in self.sel:
            for j in [0, 1, 2]:
                mol.grad[i,j] += g[l]
                l += 1

# =================================================================================================

try:
    import  pyscf.gto
    import  pyscf.dft
    import  pyscf.qmmm
    import  pyscf.grad
    import  pyscf.lib
    
    class qm3_pyscf( object ):
        def __init__( self, mol: object, 
                opts: typing.Optional[dict] = { "basis": "def2-svp", "conv_tol": 1.e-9, "charge": 0, "spin": 0,
                    "method": "b3lypg", "memory": 4096, "grid": 3, "max_cyc": 50, "nproc": 2 },
                sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
                sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
                link: typing.Optional[list] = [] ):
            if( sel_QM.sum() > 0 ):
                self.sel = numpy.argwhere( sel_QM ).ravel()
            else:
                self.sel = numpy.arange( mol.natm )
            if( sel_MM.sum() > 0 ):
                self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
            else:
                self.nbn = numpy.array( [], dtype=numpy.int32 )
            self.lnk = link[:]
            self.vla = []
            self.cx  = 1.0 / qm3.data.A0
            self.ce  = qm3.data.H2J
            self.cg  = self.ce * self.cx
    
            aQM = pyscf.gto.Mole()
            aQM.unit = "Angstrom"
            aQM.symmetry = False
            aQM.basis = opts.pop( "basis" )
            aQM.spin = opts.pop( "spin" )
            aQM.charge = opts.pop( "charge" )
            aQM.verbose = 0
            aQM.atom = ""
            for i in self.sel:
                aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                        mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
            self.vla = []
            if( len( self.lnk ) > 0 ):
                k = len( self.sel )
                for i in range( len( self.lnk ) ):
                    c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                    aQM.atom += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                    self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                    k += 1
            aQM.build()
            if( aQM.spin == 0 ):
                self.dft = pyscf.dft.RKS( aQM )
            else:
                self.dft = pyscf.dft.UKS( aQM )
            self.dft.verbose = 0
            self.dft.direct_scf = True
            self.dft.conv_tol = opts.pop( "conv_tol" )
            self.dft.max_cycle = opts.pop( "max_cyc" )
            self.dft.grids.level = opts.pop( "grid" )
            self.dft.xc = opts.pop( "method" )
            self.dft.max_memory = int( opts.pop( "memory" ) )
            if( len( self.nbn ) > 0 ):
                crd = mol.coor[self.nbn]
                crd -= mol.boxl * numpy.round( crd / mol.boxl, 0 )
                crd *= self.cx
                chg = mol.chrg[self.nbn]
                self.scf = pyscf.qmmm.mm_charge( self.dft, crd, chg, unit = "Bohr" )
            else:
                self.scf = self.dft
            pyscf.lib.num_threads( opts.pop( "nproc" ) )
    
    
        def update_coor( self, mol ):
            crd = []
            for i in self.sel:
                crd.append( mol.coor[i].tolist() )
            self.vla = []
            if( len( self.lnk ) > 0 ):
                k = len( self.sel )
                for i in range( len( self.lnk ) ):
                    c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                    crd.append( c.tolist() )
                    self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                    k += 1
            self.scf.mol.set_geom_( numpy.array( crd ) )
            if( len( self.nbn ) > 0 ):
                crd = mol.coor[self.nbn]
                crd -= mol.boxl * numpy.round( crd / mol.boxl, 0 )
                crd *= self.cx
                self.scf.mm_mol.set_geom_( crd )
    
    
        def get_func( self, mol ):
            self.update_coor( mol )
            mol.func += self.scf.kernel() * self.ce
            chg = self.scf.mulliken_pop( verbose = 0 )[1].tolist()
            k = 0
            for i in self.sel:
                mol.chrg[i] = chg[k]
                k += 1
    
    
        def get_grad( self, mol ):
            self.update_coor( mol )
            mol.func += self.scf.kernel() * self.ce
            chg = self.scf.mulliken_pop( verbose = 0 )[1].tolist()
            k = 0
            for i in self.sel:
                mol.chrg[i] = chg[k]
                k += 1
            g = self.scf.Gradients().run( grid_response = True ).de.flatten().tolist()
            Link_grad( self.vla, g )
            k = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[k] * self.cg
                    k += 1
            if( len( self.nbn ) > 0 ):
                den = self.scf.make_rdm1()
                dr  = self.scf.mol.atom_coords()[:,None,:] - self.scf.mm_mol.atom_coords()
                r   = numpy.linalg.norm( dr, axis = 2 )
                g   = numpy.einsum( "r,R,rRx,rR->Rx", self.scf.mol.atom_charges(), self.scf.mm_mol.atom_charges(), dr, r ** -3 )
                if( len( den.shape ) == 3 ):
                    for i,q in enumerate( self.scf.mm_mol.atom_charges() ):
                        with self.scf.mol.with_rinv_origin( self.scf.mm_mol.atom_coord( i ) ):
                            v = self.scf.mol.intor( "int1e_iprinv" )
                        g[i] += ( numpy.einsum( "ij,xji->x", den[0], v ) + numpy.einsum( "ij,xij->x", den[0], v.conj() ) ) * -q
                        g[i] += ( numpy.einsum( "ij,xji->x", den[1], v ) + numpy.einsum( "ij,xij->x", den[1], v.conj() ) ) * -q
                else:
                    for i,q in enumerate( self.scf.mm_mol.atom_charges() ):
                        with self.scf.mol.with_rinv_origin( self.scf.mm_mol.atom_coord( i ) ):
                            v = self.scf.mol.intor( "int1e_iprinv" )
                        g[i] += ( numpy.einsum( "ij,xji->x", den, v ) + numpy.einsum( "ij,xij->x", den, v.conj() ) ) * -q
                k = 0
                for i in self.nbn:
                    mol.grad[i,:] += g[k,:] * self.cg
                    k += 1

except:
    print( ">> PySCF not available" )

# =================================================================================================

class qm3_gaussian( object ):
    def __init__( self, mol: object, fdesc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
        self.exe = "bash r.gauss"
        self.inp = fdesc.read()
        self.gmm = ( self.inp.lower().find( "prop=(field,read)" ) > -1 )
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0


    def mk_input( self, mol, run ):
        s_qm = ""
        for i in self.sel:
            s_qm += "%2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                s_qm += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        s_mm = ""
        s_nq = ""
        for i in self.nbn:
            tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
            s_mm += "%20.10lf%20.10lf%20.10lf%12.4lf\n"%( tmp[0], tmp[1], tmp[2], mol.chrg[i] )
            s_nq += "%20.10lf%20.10lf%20.10lf\n"%( tmp[0], tmp[1], tmp[2] )
        s_rn = ""
        if( run == "grad" ):
            s_rn = "force"
#        elif( run == "hess" ):
#            s_rn = "freq=noraman cphf(maxinv=10000)"
        s_wf = ""
        if( os.access( "gauss.chk", os.R_OK ) ):
            s_wf = "guess=(read)"
        f = open( "gauss.com", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_guess", s_wf )
        buf = buf.replace( "qm3_charges", s_mm[:-1] )
        buf = buf.replace( "qm3_field", s_nq[:-1] )
        f.write( buf )
        f.write( "\n\n\n\n\n" )
        f.close()


    def parse_log( self, mol, run ):
        fd = open( "Test.FChk", "rt" )
        l = fd.readline()
        while( l != "" ):
            if( l[0:12] == "Total Energy" ):
                mol.func += float( l.split()[3] ) * self.ce
            if( run in [ "grad", "hess" ] and l[0:18] == "Cartesian Gradient" ):
                i = int( l.split()[-1] )
                j = int( i // 5 ) + ( i%5 != 0 )
                i = 0
                g = []
                while( i < j ):
                    l = fd.readline()
                    for itm in l.split():
                        g.append( float( itm ) * self.cg )
                    i += 1
                Link_grad( self.vla, g )
                k = 0
                for i in self.sel:
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += g[k]
                        k += 1
# --------------------------------------------------------------------------------
#                # read hessian (columns)
#                if( run == "hess" ):
#                    l = fd.readline()
#                    i = int( l.split()[-1] )
#                    j = int( i // 5 ) + ( i % 5 != 0 )
#                    i = 0
#                    h = []
#                    while( i < j ):
#                        l = fd.readline()
#                        for itm in l.split():
#                            h.append( float( itm ) * self.ch )
#                        i += 1
#                    # truncate LAs and swap hessian (cols>>rows)
#                    i = 3 * len( self.sel )
#                    j = i * ( i + 1 ) // 2
#                    t = qm3.maths.matrix.from_upper_diagonal_columns( h[0:j], i )
#                    for j in range( i * i ):
#                        mol.hess[j] += t[j]
# --------------------------------------------------------------------------------
            if( l[0:11] == "ESP Charges" ):
                i = int( l.split()[-1] )
                j = int( i // 5 ) + ( i % 5 != 0 )
                i = 0
                k = 0
                while( i < j ):
                    l = fd.readline()
                    for itm in l.split():
                        if( k < len( self.sel ) ):
                            mol.chrg[self.sel[k]] = float( itm )
                            k += 1
                    i += 1
            l = fd.readline()
        fd.close()
        if( len( self.nbn ) > 0 ):
            fd = open( "gauss.log", "rt" )
            l = fd.readline()
            fl = True
            while( l != "" and fl ):
                if( l[0:29] == " Self energy of the charges =" ):
                    fl = False
                    mol.func -= float( l.split()[-2] ) * self.ce
                l = fd.readline()
            if( run in [ "grad", "hess" ] and self.gmm ):
                fl = True
                while( l != "" and fl ):
                    if( l.strip() == "Potential          X             Y             Z" ):
                        fl = False
                        for i in range( 1 + len( self.sel ) + len( self.lnk ) ):
                            fd.readline()
                        for i in self.nbn:
                            t = fd.readline().split()[2:]
                            for j in [0, 1, 2]:
                                mol.grad[i,j] += - self.cg * mol.chrg[i] * float( t[j] )
                    l = fd.readline()
            fd.close()
        os.unlink( "Test.FChk" )


    def get_func( self, mol ):
        self.mk_input( mol, "ener" )
        os.system( self.exe )
        self.parse_log( mol, "ener" )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        self.parse_log( mol, "grad" )


# =================================================================================================
# orca

class qm3_orca( object ):
    def __init__( self, mol: object, fdesc,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            sel_MM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool ),
            link: typing.Optional[list] = [] ):
        if( sel_QM.sum() > 0 ):
            self.sel = numpy.argwhere( sel_QM ).ravel()
        else:
            self.sel = numpy.arange( mol.natm )
        if( sel_MM.sum() > 0 ):
            self.nbn = numpy.argwhere( numpy.logical_and( sel_MM, numpy.logical_not( sel_QM ) ) ).ravel()
        else:
            self.nbn = numpy.array( [], dtype=numpy.int32 )
        self.lnk = link[:]
        self.vla = []
        self.exe = "bash r.orca"
        self.inp = fdesc.read()
        self.ce  = qm3.data.H2J
        self.cg  = self.ce / qm3.data.A0


    def mk_input( self, mol, run ):
        s_qm = ""
        for i in self.sel:
            s_qm += "%2s%20.10lf%20.10lf%20.10lf\n"%( qm3.data.symbol[mol.anum[i]],
                    mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] )
        if( len( self.lnk ) > 0 ):
            self.vla = []
            k = len( self.sel )
            for i in range( len( self.lnk ) ):
                c, v = Link_coor( self.lnk[i][0], self.lnk[i][1], mol )
                s_qm += "%-2s%20.10lf%20.10lf%20.10lf\n"%( "H", c[0], c[1], c[2] )
                self.vla.append( ( self.sel.searchsorted( self.lnk[i][0] ), k, v ) )
                k += 1
        s_mm = ""
        if( len( self.nbn ) > 0 ):
            s_mm = "%pointcharges \"orca.pc\""
            f = open( "orca.pc", "wt" )
            f.write( "%d\n"%( len( self.nbn ) ) )
            for i in self.nbn:
                tmp = mol.coor[i] - mol.boxl * numpy.round( mol.coor[i] / mol.boxl, 0 )
                f.write( "%12.4lf%20.10lf%20.10lf%20.10lf\n"%( mol.chrg[i], tmp[0], tmp[1], tmp[2] ) )
            f.close()
        s_rn = ""
        if( run == "grad" ):
            s_rn = "engrad"
        f = open( "orca.inp", "wt" )
        buf = self.inp.replace( "qm3_atoms", s_qm[:-1] )
        buf = buf.replace( "qm3_job", s_rn )
        buf = buf.replace( "qm3_charges", s_mm )
        f.write( buf )
        f.close()


    def parse_log( self, mol, run ):
        if( run == "grad" ):
            f = open( "orca.engrad", "rt" )
            t = re.compile( "[0-9\.\-]+" ).findall( f.read() )
            f.close()
            n = int( t[0] )
            mol.func += float( t[1] ) * self.ce
            g = [ float( t[i] ) * self.cg for i in range( 2, 2 + n * 3 ) ]
            Link_grad( self.vla, g )
            k = 0
            for i in self.sel:
                for j in [0, 1, 2]:
                    mol.grad[i,j] += g[k]
                    k += 1
            if( len( self.nbn ) > 0 and os.access( "orca.pcgrad", os.R_OK ) ):
                f = open( "orca.pcgrad", "rt" )
                t = re.compile( "[0-9\.\-]+" ).findall( f.read() )
                f.close()
                n = int( t[0] )
                g = [ float( t[i] ) * self.cg for i in range( 1, 1 + n * 3 ) ]
                k = 0
                for i in self.nbn:
                    for j in [0, 1, 2]:
                        mol.grad[i,j] += g[k]
                        k += 1
        else:
            f = open( "orca.out", "rt" )
            mol.func += self.ce * float( re.compile( "FINAL SINGLE POINT ENERGY[\ ]*([0-9\.\-]+)" ).findall( f.read() )[0] )
            f.close()
        # parse orca output in search for "^CHELPG Charges" (only if chelpg is found in self.inp)
        for ff in glob.glob( "orca.*" ):
            if( ff != "orca.gbw" and ff != "orca.ges" ):
                os.unlink( ff )


    def get_func( self, mol ):
        self.mk_input( mol, "ener" )
        os.system( self.exe )
        self.parse_log( mol, "ener" )


    def get_grad( self, mol ):
        self.mk_input( mol, "grad" )
        os.system( self.exe )
        self.parse_log( mol, "grad" )

# =================================================================================================
# mmres
