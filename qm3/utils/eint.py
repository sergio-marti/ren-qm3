import  numpy
import  typing
import  qm3.data

try:
    import  qm3.utils._eint
    has_so = True
    print( ">> qm3.utils.eint: OpenCL" )
except:
    has_so = False
    print( ">> qm3.utils.eint: python" )


def read_rho_cube( fdsc: typing.IO, qm_charge = 0, skip_1s = False ):
    """
    cubegen 0 Density=SCF gauss.fchk dens.cube 0 h

    cartesian coordinates in a.u. (bohrs)
    """
    fdsc.readline()
    fdsc.readline()
    
    tokens = fdsc.readline().split()
    n_atoms = int( tokens[0] )
    origin = numpy.array( [ float( tokens[1] ), float( tokens[2] ), float( tokens[3] ) ] )
    
    tokens_x = fdsc.readline().split()
    tokens_y = fdsc.readline().split()
    tokens_z = fdsc.readline().split()
    nvoxels = numpy.array( [ int( tokens_x[0] ), int( tokens_y[0] ), int( tokens_z[0] ) ] )
    dr_x = numpy.array( [ float( tokens_x[1] ), float( tokens_x[2] ), float( tokens_x[3] ) ] )
    dr_y = numpy.array( [ float( tokens_y[1] ), float( tokens_y[2] ), float( tokens_y[3] ) ] )
    dr_z = numpy.array( [ float( tokens_z[1] ), float( tokens_z[2] ), float( tokens_z[3] ) ] )
    dV = abs( numpy.dot( dr_x, numpy.cross( dr_y, dr_z ) ) )

    out = {}
    out["atom"] = numpy.zeros( ( n_atoms, 3 ) )
    out["anum"] = numpy.zeros( n_atoms )
    for i in range( n_atoms ):
        tokens = fdsc.readline().split()
        out["atom"][i,0] = float( tokens[2] )
        out["atom"][i,1] = float( tokens[3] )
        out["atom"][i,2] = float( tokens[4] )
        if( skip_1s ):
            tmp = float( tokens[0] )
            out["anum"][i] = tmp - 2.0 if tmp > 2.0 else tmp
        else:
            out["anum"][i] = float( tokens[0] )
        
    data = numpy.fromfile( fdsc, sep = ' ', dtype = float )
    rho = data.reshape( nvoxels )
    
    ix, iy, iz = numpy.meshgrid( numpy.arange( nvoxels[0] ), numpy.arange( nvoxels[1] ), numpy.arange( nvoxels[2] ), indexing = 'ij' )
    out["coor"] = ( origin + ix[..., None] * dr_x + iy[..., None] * dr_y + iz[..., None] * dr_z )
    out["coor"] = out["coor"].reshape( -1, 3 )
    out["chrg"] = -1.0 * rho.flatten() * dV

    n_ele = numpy.sum( out["anum"] ) - qm_charge
    print( "n_elec:", n_ele )
    c_ele = - numpy.sum( out["chrg"] )
    print( "c_elec:", c_ele )
    out["chrg"] *= n_ele / c_ele
    
    return( out )



def calc_rho_eint( cube_data, coor, chrg ):
    icrd = coor / qm3.data.A0
    if( has_so ):
        e_ele = qm3.utils._eint.calc_rho_elec( cube_data["coor"], cube_data["chrg"], icrd, chrg )
        #e_ele = qm3.utils._eint.calc_rho_elec(
        #    numpy.ascontiguousarray( cube_data["coor"], dtype=numpy.float64 ),
        #    numpy.ascontiguousarray( cube_data["chrg"], dtype=numpy.float64 ),
        #    numpy.ascontiguousarray( icrd, dtype=numpy.float64 ),
        #    numpy.ascontiguousarray( chrg, dtype=numpy.float64 ) )
    else:
        # ---------------------------------------------------------------------------------------------
        #d_ele = numpy.linalg.norm( cube_data["coor"][:, None, :] - icrd[None, :, :], axis = 2 )
        #d_ele[d_ele < 1e-5] = 1e-5 
        #p_ele = numpy.sum( chrg[None, :] / d_ele, axis = 1 )
        #e_ele = numpy.sum( cube_data["chrg"] * p_ele )
        # ---------------------------------------------------------------------------------------------
        e_ele = 0.0
        for i in range( coor.shape[0] ):
            d_ele = numpy.linalg.norm( cube_data["coor"] - icrd[i], axis = 1 )
            d_ele[d_ele < 1e-5] = 1e-5
            e_ele += chrg[i] * numpy.sum( cube_data["chrg"] / d_ele )
    # ---------------------------------------------------------------------------------------------
    d_nuc = numpy.linalg.norm( cube_data["atom"][:, None, :] - icrd[None, :, :], axis = 2 )
    p_nuc = numpy.sum( chrg[None, :] / d_nuc, axis = 1)
    e_nuc = numpy.sum( cube_data["anum"] * p_nuc )
    return( ( e_ele + e_nuc ) * qm3.data.H2J )


