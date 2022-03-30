import  numpy
import  typing
import  qm3
import  qm3.engines.mopac
import  qm3.utils
import  qm3.utils.hessian
import  qm3.actions.minimize
import  qm3.actions.paths
import  qm3.actions.ring_polymer
import  io
import  pickle
import  os

mol = qm3.molecule()
f = io.StringIO("""15

C          -2.5226803073       -0.7246958984        0.1601445509
C          -1.1995765456       -0.0535287046        0.0755155053
H          -3.2981239394       -0.0243369443        0.5577546376
H          -2.4580100512       -1.6235101367        0.8213127507
H          -2.8296073276       -1.0542085116       -0.8652309031
C          -0.0007797913       -0.7870809465        0.0055250622
C           1.1978437126       -0.0554069219       -0.0843287564
H          -0.0006663114       -1.8787673288        0.0203154670
C           2.5211253331       -0.7282609838       -0.1507275223
H           2.4560792403       -1.6463518948       -0.7848071286
H           2.8295317028       -1.0269760408        0.8836082128
H           3.2957339499       -0.0396478087       -0.5699261398
O           1.1726410920        1.2459916150       -0.1026088826
O          -1.1746864683        1.2478908060        0.0585215393
H          -0.0010757399        1.6179486274       -0.0270720941
""" ) 
mol.xyz_read( f ) 
mol.guess_atomic_numbers()
mol.engines["eqm"] = qm3.engines.mopac.run( mol, "AM1", 0 )


def calc_hess( self: object, step: int, raise_RT: typing.Optional[bool] = False ):
    self.hess = qm3.utils.hessian.numerical( self )
    self.get_grad()
    if( raise_RT ):
        return( qm3.utils.hessian.raise_RT( self.hess, qm3.utils.RT_modes( self ) ) )
    else:
        return( self.hess )


if( not os.path.isfile( "rpi_T.pk" ) ):
    qm3.actions.minimize.baker( mol,
            lambda obj,stp: calc_hess( obj, stp, True ),
            step_number = 100, print_frequency = 1, follow_mode = 0, gradient_tolerance = 1.0 )
    mol.xyz_write( open( "xyz", "wt" ) )
    with open( "rpi_T.pk", "wb" ) as f:
        pickle.dump( mol.coor, f )
        pickle.dump( mol.func, f )
        pickle.dump( mol.hess, f )


if( not os.path.isfile( "rpi_R.pk" ) ):
    qm3.actions.paths.page_mciver( mol, calc_hess,
            step_number = 100, print_frequency = 10 )
    qm3.actions.minimize.baker( mol,
            lambda obj,stp: calc_hess( obj, stp, True ),
            step_number = 100, print_frequency = 1, follow_mode = -1, gradient_tolerance = 1.0 )
    with open( "rpi_R.pk", "wb" ) as f:
        pickle.dump( mol.coor, f )
        pickle.dump( mol.func, f )
        pickle.dump( mol.hess, f )


obj = qm3.actions.ring_polymer.instanton( mol, calc_hess, nbeads = 32 )

with open( "rpi_R.pk", "rb" ) as f:
    r_coor = pickle.load( f )
    r_func = pickle.load( f )
    r_hess = pickle.load( f )
    print( "R_ener: ", r_func )

with open( "rpi_T.pk", "rb" ) as f:
    t_coor = pickle.load( f )
    t_func = pickle.load( f )
    t_hess = pickle.load( f )
    print( "T_ener: ", t_func )

obj.calc_TST( r_coor, r_func, r_hess, t_coor, t_func, t_hess, 1.0, 2.0 )
obj.setup()

if( not os.path.isfile( "rpi.pk" ) ):
    qm3.actions.minimize.baker( obj, lambda obj,stp: obj.get_hess( obj, 0 ),
            step_size = 0.1, step_number = 100, print_frequency = 1, follow_mode = 0, gradient_tolerance = 1.0 )
    with open( "rpi.pk", "wb" ) as f:
        pickle.dump( obj.func, f )
        pickle.dump( obj.coor, f )
        pickle.dump( obj.ener, f )
        pickle.dump( obj.hess, f )

with open( "rpi.pk", "rb" ) as f:
    obj.func = pickle.load( f )
    obj.coor = pickle.load( f )
    obj.ener = pickle.load( f )
    obj.hess = pickle.load( f )

obj.calc_RPI( r_coor, r_func, r_hess )
obj.plot( r_func, t_func )

print( "\n\n\n>> ./eckart_bell.py -355.886 -268.853 -355.886 1894.6 300" )
