#!/usr/bin/env python3
import  typing
import  numpy
import  torch
import  torch.nn
import  torch.optim


def xcoul_info( crd: torch.tensor, env: torch.tensor ) -> torch.tensor:
    b_size, n_atoms, _ = crd.shape
    siz = env.shape[1]
    out = torch.zeros( ( b_size, n_atoms, siz * siz ), dtype=crd.dtype, device=crd.device )
    for i in range( n_atoms ):
        dr = crd[:, env[i]] - crd[:, i].unsqueeze( 1 )
        r2 = torch.sum( dr ** 2, dim = 2 )
        tmp = torch.zeros( ( b_size, siz, 4 ), dtype=crd.dtype, device=crd.device )
        tmp[:, :, 0] = 1.0 / torch.sqrt( r2 )
        tmp[:, :, 1:4] = dr / r2.unsqueeze( -1 )
        out[:, i, :] = torch.einsum( "bij,bjk->bik", tmp, tmp.transpose( 1, 2 ) ).reshape( b_size, -1 )
    return( out )

    
def ebuild( crd: torch.tensor, chg: torch.tensor, n_dst: int, n_pot: typing.Optional[int] = 0 ):
    b_size, n_atoms, _ = crd.shape
    env = []
    dist = torch.norm( crd.unsqueeze( 2 ) - crd.unsqueeze( 1 ), dim = 3 ) + 1e-10
    avr_dst = dist.mean( dim = 0 )
    avr_pot = torch.abs( ( chg / dist ).mean( dim = 0 ) )
    n_tot = n_dst + n_pot
    for i in range( n_atoms ):
        nearest = set( torch.argsort( avr_dst[i] )[1:n_dst+1].tolist() )
        pot_srt = torch.argsort( avr_pot[i], descending = True )
        pot_srt = [ j.item() for j in pot_srt if j != i ]
        for j in pot_srt:
            if( len( nearest ) >= n_tot ):
                break
            nearest.add( j )
        env.append( sorted( nearest ) )
    return( torch.tensor( env, dtype=torch.int32 ) )


class ml_atom( torch.nn.Module ):
    def __init__( self, knd: str, net: list ):
        super().__init__()
        self.name  = f"atom_{knd}.pth"
        mod = []
        for i in range( len( net ) - 1 ):
            mod.append( torch.nn.Linear( net[i], net[i+1] ) )
            mod.append( torch.nn.Tanh() )
        mod.append( torch.nn.Linear( net[-1], 1 ) )
        self.model = torch.nn.Sequential( *mod )
    
    def forward( self, x: torch.tensor ) -> torch.tensor:
        return( self.model( x ) )

    def save( self ):
        torch.save( self.state_dict(), self.name  )

    def load( self, device: str ):
        self.load_state_dict( torch.load( self.name, map_location=torch.device( device ) ) )
        self.eval()


class run( object ):
    def __init__( self, xref: numpy.array, eref: numpy.array, desc: torch.tensor,
                        sele: numpy.array, labl: list, netw: list, device: str ):
        self.dev = device
        self.ref = xref.copy()
        self.dsp = eref.copy()
        self.env = desc
        self.sel = numpy.flatnonzero( sele )
        self.lbl = labl[:]
        self.knd = list( set( labl ) )
        self.net = { i: ml_atom( i, netw ) for i in self.knd }
        print( "Erng:", self.dsp )
        print( "Kind:", self.knd )
        print( "Atom:", self.lbl )
        for k in self.knd:
            self.net[k].to( device )
        print( self.net[self.knd[0]].model )

    def parameters( self ) -> list:
        return( [ p for k in self.knd for p in self.net[k].parameters() ] )

    def save( self ):
        for k in self.knd:
            self.net[k].save()

    def load( self ):
        for k in self.knd:
            self.net[k].load( self.dev )

    def __call__( self, inp: torch.tensor ) -> torch.tensor:
        out = self.net[self.lbl[0]]( inp[:,0,:] )
        for i in range( 1, len( self.lbl ) ):
            out += self.net[self.lbl[i]]( inp[:,i,:] )
        return( out )

    def get_func( self, mol ) -> float:
        crd = torch.tensor( mol.coor[self.sel], dtype=torch.float32, device=self.dev ).unsqueeze( 0 )
        inp = xcoul_info( crd, self.env )
        tmp = ( self.dsp[1] - self.dsp[0] ) / 2.0
        out = ( float( self( inp ).cpu().numpy().ravel()[0] ) + 1.0 ) * tmp + self.dsp[0]
        mol.func += out
        return( out )

    def get_grad( self, mol ) -> float:
        crd = mol.coor[self.sel]
        crd -= numpy.mean( crd, axis = 0 )
        cov = numpy.dot( crd.T, self.ref )
        r1, s, r2 = numpy.linalg.svd( cov )
        if( numpy.linalg.det( cov ) < 0 ):
            r2[2,:] *= -1.0
        mat = numpy.linalg.inv( numpy.dot( r1, r2 ) )
        crd = torch.tensor( crd, dtype=torch.float32, device=self.dev ).unsqueeze( 0 )
        crd.requires_grad = True
        inp = xcoul_info( crd, self.env )
        tmp = ( self.dsp[1] - self.dsp[0] ) / 2.0
        out = self( inp )
        grd = torch.autograd.grad( out.sum(), crd )[0]
        out = ( float( out.detach().cpu().numpy().ravel()[0] ) + 1.0 ) * tmp + self.dsp[0]
        mol.func += out
        mol.grad[self.sel] += numpy.dot( grd.detach().cpu().numpy()[0] * tmp, mat )
        return( out )


class scheduler:
    def __init__( self, optimizer: torch.optim.Optimizer,
                 min_lr: float, max_lr: float, steps_per_epoch: int, 
                 lr_decay: typing.Optional[int] = 1,
                 cycle_length: typing.Optional[int] = 10,
                 mult_factor: typing.Optional[int] = 2 ):
        """
            optimizer: PyTorch optimizer (e.g., torch.optim.Adam)
            min_lr: Minimum learning rate.
            max_lr: Initial maximum learning rate.
            steps_per_epoch: Number of batches per epoch.
            lr_decay: Decay factor for max_lr after each cycle.
            cycle_length: Initial number of epochs per cycle.
            mult_factor: Factor to scale cycle length after each cycle.
        """
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.batch_since_restart = 0
        self.next_restart = cycle_length * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length * steps_per_epoch
        self.mult_factor = mult_factor
        self.best_weights = None

    def clr( self ):
        fraction_to_restart = self.batch_since_restart / self.cycle_length
        lr = self.min_lr + 0.5 * ( self.max_lr - self.min_lr ) * ( 1 + numpy.cos( fraction_to_restart * numpy.pi ) )
        return( lr )

    def step( self ):
        self.batch_since_restart += 1
        lr = self.clr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        if self.batch_since_restart >= self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = int( self.cycle_length * self.mult_factor )
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay

    def get_lr( self ):
        return( [ param_group["lr"] for param_group in self.optimizer.param_groups ] )


