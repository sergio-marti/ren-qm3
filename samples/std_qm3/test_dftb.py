import  numpy
import	qm3.mol
import  qm3.fio.xplor
import  qm3.engines.dftb
import  io


mol = qm3.mol.molecule( "../charmm.pdb" )
qm3.fio.xplor.psf_read( mol, "../charmm.psf" )
mol.guess_atomic_numbers()
print( mol.anum[0:3], mol.anum[-3:] )
print( mol.chrg[0:3], mol.chrg[-3:] )

sqm = [ mol.indx["A"][1][a] for a in ['N1', 'C2', 'C3', 'N4', 'C5', 'H7', 'H8', 'C10', 'H16', 'H17', 'H18', 'H19'] ]
smm = mol.sph_sel( sqm, 12 )
print( len( sqm ), len( smm ) )
sla = [ [ mol.indx["A"][1]["C10"], mol.indx["A"][1]["C6"] ] ]
smm = list( set( smm ).difference( set( sqm + sum( sla, [] ) ) ) )

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
eng = qm3.engines.dftb.run_dynlib( mol, f, sqm, smm, sla )

mol.func = 0
mol.grad = [ 0.0 for i in range( 3 * mol.natm ) ]
eng.get_grad( mol )

print( mol.func )
print( numpy.linalg.norm( numpy.array( mol.grad ) ) )
w = mol.indx["A"][1]["C10"]
print( numpy.linalg.norm( numpy.array( mol.grad[3*w:3*w+3] ) ) )
