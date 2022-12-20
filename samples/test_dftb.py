import	numpy
import	qm3
import  qm3.engines.dftb
import  io
import  os
import  sys

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep

mol = qm3.molecule()
mol.pdb_read( open( cwd + "charmm.pdb" ) )
mol.psf_read( open( cwd + "charmm.psf" ) )
mol.guess_atomic_numbers()
print( mol.anum )
print( mol.chrg )

sqm = mol.resn == "WAT"
for a in [ "C6", "C9", "H11", "H12", "H13", "H14", "H15" ]:
    sqm[mol.indx["A"][1][a]] = True
sqm = numpy.logical_not( sqm )
smm = mol.sph_sel( sqm, 12 )
#sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ) ]
sla = [ ( mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"], [ mol.indx["A"][1]["H11"], mol.indx["A"][1]["H12"] ] ) ]
print( sqm.sum(), smm.sum() )

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
    Prefix = "%s/3ob-3-1/"
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
"""%( cwd ) )

mol.engines["qm"] = qm3.engines.dftb.run( mol, f, sqm, smm, sla )

mol.get_grad()
#print( round( mol.func, 1 ), "/ -36796.1" )
#print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 560.9" )
#print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 61.9" )
print( round( mol.func, 1 ), "/ -36826.6" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 556.5" )
print( round( numpy.linalg.norm( mol.grad[mol.indx["A"][1]["C10"]] ), 1 ), "/ 59.9" )
