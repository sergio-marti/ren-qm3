import	numpy
import	qm3
import  qm3.engines.dftb
import  io


mol = qm3.molecule()
mol.pdb_read( open( "charmm.pdb" ) )
mol.psf_read( open( "charmm.psf" ) )
mol.guess_atomic_numbers()
print( mol.anum )
print( mol.chrg )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]

f = io.StringIO( """
Driver = {}
Geometry = GenFormat {
qm3_atoms
}
Hamiltonian = DFTB {
  SCC = Yes
  MaxSCCIterations = 1000
  Mixer = DIIS {}
  SlaterKosterFiles = Type2FileNames {
    Prefix = "./3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
  }
  MaxAngularMomentum { H = "s"; C = "p"; N = "p" }
  Charge = 1
  ThirdOrderFull = Yes
  HubbardDerivs = { 
    H = -0.1857
    C = -0.1492
    N = -0.1535
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

mol.engines.append( qm3.engines.dftb.run( mol, f, sqm, smm, sla ) )

mol.get_grad()
print( mol.func )
assert( numpy.fabs( mol.func - -36796.061200114185 ) < 1.e-4 ), "DFTB+: function error"
print( numpy.linalg.norm( mol.grad ) )
assert( numpy.fabs( numpy.linalg.norm( mol.grad ) - 560.9451224737893 ) < 1.e-4 ), "DFTB+: gradient error"
