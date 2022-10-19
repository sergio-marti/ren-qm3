import  numpy
import  typing
import  qm3.engines._ml_info


class network( object ):
    """
    ===  Sequential Softplus-activated Dense Layers  ===

    import  tensorflow as tf

    ref = np.min( out )
    out -= ref
    mod = tf.keras.models.Sequential( [ 
        tf.keras.layers.Dense( 256, activation = "softplus" ),
        tf.keras.layers.Dense( 128, activation = "softplus" ),
        tf.keras.layers.Dense(  64, activation = "softplus" ),
        tf.keras.layers.Dense(  32, activation = "softplus" ),
        tf.keras.layers.Dense(   1, activation = "softplus" ) ] )
    mod.compile( optimizer = tf.keras.optimizers.Adam( learning_rate = 0.0001 ),
        loss = "MeanSquaredError" )
    stp = tf.keras.callbacks.ModelCheckpoint( "model.h5", monitor = "loss",
        verbose = 1, mode = "min", save_best_only = True )
    mod.fit( inp, out, batch_size = 2048, epochs = 10000, callbacks = [ stp ], verbose = 0 )

    mod = tf.keras.models.load_model( "model.h5" )
    obj = qm3.engines.delta_ml.network()
    obj.coef = [ m.numpy() for m in mod.trainable_weights ]
    obj.oref = float( ref )
    """
    def __init__( self ):
        self.coef = []
        self.oref = 0.0


    @staticmethod
    def f_actv( vec: numpy.array ) -> numpy.array:
        # math.log( sys.float_info.max )
        return( numpy.log( 1.0 + numpy.exp( numpy.where( vec > 709.7827, 709.7827, vec ) ) ) )


    @staticmethod
    def g_actv( vec: numpy.array ) -> numpy.array:
        # math.log( sys.float_info.max )
        return( 1.0 / ( 1.0 + numpy.exp( numpy.where( vec < -709.7827, -709.7827, - vec ) ) ) )


    # -- scalar
    def get_func( self, data: numpy.array ) -> float:
        out = data.reshape( ( 1, len( data ) ) )
        for i in range( 0, len( self.coef ), 2 ):
            out = self.f_actv( numpy.dot( out, self.coef[i] ) + self.coef[i+1] )
        return( float( out ) + self.oref )


    # -- scalar, input dimension
    def get_grad( self, data: numpy.array ) -> tuple:
        inp = data.reshape( ( 1, len( data ) ) )
        f_tmp = numpy.dot( inp, self.coef[0] ) + self.coef[1]
        g_tmp = self.coef[0] * self.g_actv( f_tmp )
        for i in range( 2, len( self.coef ), 2 ):
            f_tmp = numpy.dot( self.f_actv( f_tmp ), self.coef[i] ) + self.coef[i+1]
            g_tmp = numpy.dot( g_tmp, self.coef[i] * self.g_actv( f_tmp ) )
        return( float( self.f_actv( f_tmp ) ) + self.oref, g_tmp.ravel() )

# =================================================================================================

try:
    import  tensorflow

    class tf_network( object ):
        def __init__( self, model: str ):
            self.mode = tensorflow.keras.models.load_model( model )
            self.oref = 0.0


        def get_func( self, data: numpy.array ) -> float:
            return( float( self.mode( data.reshape( ( 1, len( data ) ) ), training = False ) ) + self.oref )


        def get_grad( self, data: numpy.array ) -> tuple:
            inp = tensorflow.convert_to_tensor( data.reshape( ( 1, len( data ) ) ) )
            with tensorflow.GradientTape() as grd:
                grd.watch( inp )
                lss = self.mode( inp )
            ene = float( self.mode( inp, training = False ) ) + self.oref
            return( ene, grd.gradient( lss, inp ).numpy().ravel() )
except:
    pass

# =================================================================================================

class template( object ):
    def __init__( self, network: object, mol: object, sele: numpy.array ):
        self.netw = network
        self.sele = numpy.argwhere( sele.ravel() ).ravel()


    def get_info( self, mol: object ) -> numpy.array:
        raise( NotImplementedError )


#    def get_jaco( self, mol: object ) -> numpy.array:
#        raise( NotImplementedError )


    def get_func( self, mol: object ):
        mol.func += self.netw.get_func( self.get_info( mol ) )


#    def get_grad( self, mol: object ):
#         etmp, gtmp = self.netw.get_grad( self.get_info( mol ) )
#         grd = numpy.dot( gtmp.T, self.get_jaco( mol ) )
#         grd.shape = ( len( self.sele ), 3 )
#         mol.grad[self.sele] += grd


    def get_grad( self, mol: object, disp: typing.Optional[float] = 1.e-3 ):
        mol.func += self.netw.get_func( self.get_info( mol ) )
        for i in self.sele:
            for j in [0, 1, 2]:
                bak = mol.coor[i,j]
                mol.coor[i,j] = bak + disp
                ffw = self.netw.get_func( self.get_info( mol ) )
                mol.coor[i,j] = bak - disp
                bbw = self.netw.get_func( self.get_info( mol ) )
                mol.grad[i,j] += ( ffw - bbw ) / ( 2.0 * disp )
                mol.coor[i,j] = bak

# =================================================================================================

class coulomb( template ):
    def __init__( self, network: object, mol: object, sele: numpy.array, anum: typing.Optional[bool] = False ):
        template.__init__( self, network, mol, sele )
        self.anum = []
        if( anum ):
            self.anum = mol.anum[self.sele]
        else:
            self.anum = numpy.ones( len( self.sele ) )


    def get_info( self, mol ) -> numpy.array:
        return( qm3.engines._ml_info.coul_info( self.anum, mol.coor[self.sele] ) )


#    def get_jaco( self, mol ) -> numpy.array:
#        return( qm3.engines._ml_info.coul_jaco( self.anum, mol.coor[self.sele] ) )

# =================================================================================================

# [10.1063/1.3553717]
class acsf( template ):
    def __init__( self, network: object, mol: object, sele: numpy.array ):
        template.__init__( self, network, mol, sele )


    def setup( self,
            cutx: typing.Optional[float] = 4.0,
            eta2: typing.Optional[list] = [ 1.0 ],
            dse5: typing.Optional[float] = 1.0,
            eta5: typing.Optional[list] = [ 0.1 ] ):
#            cutx: typing.Optional[float] = 6.0,
#            eta2: typing.Optional[list] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
#            dse5: typing.Optional[float] = 1.0,
#            eta5: typing.Optional[list] = [0.1 , 0.11, 0.12, 0.13, 0.14] ):
        self.cutx = cutx
        self.eta2 = eta2
        self.dse5 = dse5
        self.eta5 = eta5


    def get_info( self, mol ) -> numpy.array:
        return( qm3.engines._ml_info.acsf_info( self.cutx, self.eta2, self.dse5, self.eta5, mol.coor[self.sele] ) )
