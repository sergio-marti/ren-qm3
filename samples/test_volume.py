import  sys
import	numpy
import	qm3
import  qm3.data
import  qm3.utils._volume
import  struct
import  os

 
cwd = os.path.abspath( os.path.dirname( sys.argv[0] ) ) + os.sep


mol = qm3.molecule()

mol.pdb_read( open( cwd + "amber.pdb" ) )
mol.boxl = numpy.array( [ 25.965, 29.928, 28.080 ] )
mol.prmtop_read( open( cwd + "amber.prmtop" ) )

sus = mol.resn == "SUS"
wat = numpy.logical_not( sus )

ans = qm3.utils._volume.surface( qm3.data.r_vdw[mol.anum[sus]], mol.coor[sus] )
print( ans.shape, ans[0], ans[-1] )
print( 100 * "-" )

ans = qm3.utils._volume.molecular( 2, qm3.data.r_vdw[mol.anum[sus]], mol.coor[sus], 0.02, True )
os.rename( "volume.pdb", "volume_mol.pdb" )
print( ans )
print( 100 * "-" )

ans = qm3.utils._volume.cavity( qm3.data.r_vdw[mol.anum[wat]], mol.coor[wat],
    numpy.mean( mol.coor[sus], axis = 0 ),
    0.1, 1.4, True, False )
os.rename( "volume.pdb", "volume_acs.pdb" )
print( ans )
print( 100 * "-" )

with open( "within", "wb" ) as f:
    f.write( struct.pack( "i", numpy.sum( sus ) ) )
    for i in numpy.flatnonzero( sus ):
        for j in [0, 1, 2]:
            f.write( struct.pack( "d", mol.coor[i,j] ) )
        f.write( struct.pack( "d", 3 * qm3.data.r_vdw[mol.anum[i]] ) )

ans = qm3.utils._volume.cavity( qm3.data.r_vdw[mol.anum[wat]], mol.coor[wat],
    numpy.mean( mol.coor[sus], axis = 0 ),
    0.1, 1.4, True, True )
os.rename( "volume.pdb", "volume_trn.pdb" )
print( ans )
