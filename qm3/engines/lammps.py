import  numpy
import  typing
import  qm3.data

import  lammps
import  ctypes


class run( object ):
    def __init__( self, config = "lammps.inp", options = [ "-sc", "none" ] ):
        self.lmp = lammps.lammps( cmdargs = options )
        self.lmp.file( config )
#        self.chg = self.lmp.gather_atoms( "q", 1, 1 )
#        self.crd = self.lmp.gather_atoms( "x", 1, 3 )


    def stop( self ):
        self.lmp.close()


    def update_chrg( self, mol: object ):
#        for i in range( mol.natm ):
#            self.chg[i] = mol.chrg[i]
#        self.lmp.scatter_atoms( "q", 1, 1, self.chg )
        self.lmp.scatter_atoms( "x", 1, 3, mol.chrg.ctypes.data_as( ctypes.POINTER( ctypes.c_float ) ) )


    def update_coor( self, mol: object ):
#        k = 0
#        for i in range( mol.natm ):
#            for j in [0, 1, 2]:
#                self.crd[k] = mol.coor[i,j]
#                k += 1
#        self.lmp.scatter_atoms( "x", 1, 3, self.crd )
        self.lmp.scatter_atoms( "x", 1, 3, mol.coor.ctypes.data_as( ctypes.POINTER( ctypes.c_float ) ) )



    def get_func( self, mol: object ):
        self.update_coor( mol )
        self.lmp.command( "run 0" )
        mol.func += self.lmp.get_thermo( "pe" ) * qm3.data.K2J


    def get_grad( self, mol: object ):
        self.get_func( mol )
#        frz = self.lmp.gather_atoms( "f", 1, 3 )
#        k = 0
#        for i in range( mol.natm ):
#            for j in [0, 1, 2]:
#                mol.grad[i,j] -= frz[k] * qm3.data.K2J
#                k += 1
        mol.grad -= numpy.array( self.lmp.gather_atoms( "f", 1, 3 ) ).reshape( ( mol.natm, 3 ) ) * 4.184
