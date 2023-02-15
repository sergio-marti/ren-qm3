import  math
import  numpy
import  typing
import  qm3.data
import  plumed

#
# src/core/PlumedMainMap.inc
#

class run:
    def __init__( self, mol ):
        self.eng = plumed.Plumed()
        self.eng.cmd( "setRealPrecision", 8 )
        self.eng.cmd( "setMDLengthUnits", 0.1 )
        self.eng.cmd( "setMDTimeUnits", 0.001 )
        self.eng.cmd( "setTimestep", 1.0 )
        self.eng.cmd( "setMDChargeUnits", qm3.data.EV )
        self.eng.cmd( "setPlumedDat", "plumed.dat" )
        self.eng.cmd( "setNatoms", mol.natm )
        self.eng.cmd( "setMDEngine", "python" )
        self.eng.cmd( "setLogFile", "plumed.log" )
        self.eng.cmd( "setNoVirial" )
        self.eng.cmd( "init" )
        self.step = 0


    def get_grad( self, mol ):
        self.eng.cmd( "setStep", self.step )
        self.eng.cmd( "setBox", numpy.eye( 3 ) * mol.boxl )
        self.eng.cmd( "setPositions", mol.coor )
        self.eng.cmd( "setMasses", mol.chrg )
        frz = numpy.zeros( ( mol.natm, 3 ) )
        self.eng.cmd( "setForces", frz )
        self.eng.cmd( "calc" )
#        self.eng.cmd( "prepareCalc" )
#        self.eng.cmd( "performCalc" )
        ene = numpy.zeros( ( 1, ) )
        self.eng.cmd( "getBias", ene )
        mol.func += float( ene )
        mol.grad -= frz
        self.step += 1
        return( float( ene ) )


    def stop( self ):
        self.eng.finalize()
