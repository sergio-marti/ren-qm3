#!/usr/bin/env python3
import  sys
import	numpy
import	qm3
import  pickle
import  dftb
import  io


f = io.StringIO( """6

O       0.12109      0.06944     -0.22458
H      -0.52694      0.16499     -0.94583
H       0.70159     -0.63565     -0.54677
O      -0.45114      1.12675      2.21102
H      -0.29157      0.59483      1.39876
H       0.05804      1.92714      2.01036
""" )

mol = qm3.molecule()
mol.xyz_read( f )
mol.guess_atomic_numbers()
mol.chrg = [ -0.834, 0.417, 0.417, -0.834, 0.417, 0.417 ]

sqm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
sqm[[0,1,2]] = True
smm = numpy.zeros( mol.natm, dtype=numpy.bool_ )
smm[[3,4,5]] = True

f = io.StringIO( """
Driver = {}
Geometry = GenFormat {
qm3_atoms
}
Hamiltonian = DFTB {
  SCC = Yes
  ConvergentSccOnly = Yes
  MaxSCCIterations = 1000
  Mixer = DIIS {}
  SlaterKosterFiles = Type2FileNames {
    Prefix = "./3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
  }
  MaxAngularMomentum { H = "s"; O = "p" }
  Charge = 0
  ThirdOrderFull = Yes
  HubbardDerivs = { 
    H = -0.1857
    O = -0.1575
  }
  HCorrection = Damping { Exponent = 4.0 }
qm3_guess
  ElectricField = {
    PointCharges = {
      CoordsAndCharges [Angstrom] = DirectRead {
        Records = qm3_nchg
        File = "charges.dat"
      }
    }
  }
}
Options { WriteDetailedOut = Yes }
Analysis {
  MullikenAnalysis = Yes
  WriteBandOut = No
qm3_job
}
ParserOptions { WriteHSDInput = No }
""" )
eqm = dftb.run( mol, f, sel_QM = sqm, sel_MM = smm )
mol.func = 0.0
eqm.get_func( mol )
print( mol.func )
print()

bak = mol.chrg.copy()

for i in range( 4 ):
    mol.func = 0.0
    mol.chrg = numpy.zeros( mol.natm, dtype=numpy.float64 )
    eqm.get_func( mol, -1 )
    print( mol.func )
    print()

    mol.func = 0.0
    mol.chrg = bak.copy()
    eqm.get_func( mol, -1 )
    print( mol.func )
    print()
