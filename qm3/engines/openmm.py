import  numpy
import  typing
import  qm3.data

import  openmm
import  openmm.app
import  openmm.unit
import  time


class run( object ):
    def __init__( self, omm_sys: object, omm_top: object,
            sel_QM: typing.Optional[numpy.array] = numpy.array( [], dtype=numpy.bool_ ),
            platform = "CPU" ):
        nqm = sel_QM.sum()
        if( nqm > 0 ):
            idx = numpy.argwhere( sel_QM.ravel() ).ravel()
            msk = sel_QM * 1
            for ii in range( omm_sys.getNumForces() ):
                cur = omm_sys.getForce( ii )
                if( type( cur ) == openmm.NonbondedForce ):
                    for i in range( 0, nqm - 1 ):
                        for j in range( i + 1, nqm ):
                            cur.addException( idx[i], idx[j], 0.0, 0.0, 0.0, replace = True )
                    for i in idx:
                        tmp = cur.getParticleParameters( i )
                        cur.setParticleParameters( i, 0.0, tmp[1], tmp[2] )
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
                        print( ">> there are charmm-cmaps defined... (and unhandled!)" )
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
                tmp = self.nbn.getParticleParameters( i )
                self.nbn.setParticleParameters( i, mol.chrg[i], tmp[1], tmp[2] )
            self.nbn.updateParametersInContext( self.sim.context )
        else:
            print( ">> Unable to update charges: no openmm.NonbondedForce available!" )


    def update_coor( self, mol: object ):
#        tmp = []
#        for i in range( mol.natm ):
#            tmp.append( openmm.Vec3( mol.coor[i,0], mol.coor[i,1], mol.coor[i,2] ) * openmm.unit.angstrom )
#        self.sim.context.setPositions( tmp )
        self.sim.context.setPositions( mol.coor * 0.1 )


    def get_func( self, mol: object ):
        self.update_coor( mol )
        stt = self.sim.context.getState( getEnergy = True, getForces = False )
        mol.func += stt.getPotentialEnergy().value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )


    def get_grad( self, mol: object ):
        self.update_coor( mol )
        stt = self.sim.context.getState( getEnergy = True, getForces = True )
        mol.func += stt.getPotentialEnergy().value_in_unit( openmm.unit.kilojoule/openmm.unit.mole )
#        frc = stt.getForces()
#        for i in range( mol.natm ):
#            for j in [0, 1, 2]:
#                mol.grad[i,j] -= frc[i][j].value_in_unit( openmm.unit.kilojoule/(openmm.unit.angstrom*openmm.unit.mole) )
        frc = numpy.array( stt.getForces( True ) )
        mol.grad -= frc * 0.1
