import  numpy
import  typing
import  qm3.data
import  sander
import  scipy.io



def coordinates_read( mol: object, fdsc: typing.IO ):
    fdsc.readline()
    nat = int( fdsc.readline().strip() )
    if( mol.natm == nat ):
        crd = []
        n3  = nat * 3
        while( len( crd ) < n3 ):
            crd += [ float( j ) for j in fdsc.readline().split() ]
        mol.coor = numpy.array( crd ).reshape( ( mol.natm, 3 ) )
        mol.boxl = numpy.array( [ float( j ) for j in fdsc.readline().split()[0:3] ] )



def coordinates_write( mol, fdsc: typing.IO ):
    fdsc.write( "default_name\n%6d\n"%( mol.natm ) )
    c = 0
    for i in range( mol.natm ):
        for j in [0, 1, 2]:
            fdsc.write( "%12.7lf"%( mol.coor[i,j] ) )
            c += 1
            if( c % 6 == 0 ):
                fdsc.write( "\n" )
    if( c % 6 != 0 ):
        fdsc.write( "\n" )
    fdsc.write( "%12.7lf%12.7lf%12.7lf%12.7lf%12.7lf%12.7lf\n"%( mol.boxl[0], mol.boxl[1], mol.boxl[2], 90.0, 90.0, 90.0 ) )



class netcdf( object ):
    def __init__( self, fname: str ):
        self.fdsc = scipy.io.netcdf_file( fname, "r" )
        self.coor = self.fdsc.variables["coordinates"]
        self.boxl = None
        if( "cell_lengths" in self.fdsc.variables ):
            self.boxl = self.fdsc.variables["cell_lengths"]
        self.mult = len( self.coor.shape ) == 3
        self.cfrm = 0
        print( "* [%s] "%( fname ) + ( 55 - len( fname ) ) * "-" )
        if( self.mult ):
            print( "+ \tNFrames: %ld"%( self.coor.shape[0] ) )
            print( "+ \tAtoms: %ld"%( self.coor.shape[1] ) )
        else:
            print( "+ \tNFrames: %ld"%( 1 ) )
            print( "+ \tAtoms: %ld"%( self.coor.shape[0] ) )
        print( 60 * "-" )


    def goto( self, frame: int ):
        if( frame >= 0 and frame < self.coor.shape[0] and self.mult ):
            self.cfrm = frame


    def next( self, mol: object ) -> bool:
        out = False
        if( self.mult ):
            if( self.cfrm >= 0 and self.cfrm < self.coor.shape[0] and self.coor.shape[1] == mol.natm ):
                out = True
                mol.coor = self.coor.data[self.cfrm].copy()
                if( self.boxl != None ):
                    mol.boxl = self.boxl.data[self.cfrm].copy()
                self.cfrm += 1
        else:
            if( self.coor.shape[0] == mol.natm ):
                mol.coor = self.coor.data.copy()
                if( self.boxl != None ):
                    mol.boxl = self.boxl.data.copy()
        return( out )


    def close( self ):
        del( self.coor )
        if( self.boxl != None ):
            del( self.boxl )
        self.fdsc.close()
          


class run( object ):
    def __init__( self, prmtop: str, mol: object,
                        qm_mask: typing.Optional[str] = "",
                        qm_meth: typing.Optional[str] = "AM1",
                        qm_chrg: typing.Optional[int] = 0,
                        qm_cut: typing.Optional[float] = 12.0 ):
        """
        Amber masks:

        :1-10       "residues 1 to 10"
        :1,3,5      "residues 1, 3, and 5"

        :LYS        "all lysine residues"
        :ARG,ALA    "all arginine and alanine residues

        @12,17      "atoms 12 and 17"
        @54-85      "all atoms from 54 to 85"

        @CA         "all atoms with name CA"
        @CA,C,N     "all atoms with names CA or C or N"
        @X=         "all atoms starting with X"

        <:X         "all residues within X Angs"
        <@X         "all atoms    within X Angs"

        &           AND
        |           OR
        !           NOT


        @C= & ! @CA         "all atoms starting with C but not named CA"
        (:1 <:3.0) & :WAT   "all water molecules within 3 Angs from residue 1"

        """
        if( qm_mask == "" ):
            self.obj = sander.setup( prmtop,
                mol.coor.ravel(), [ mol.boxl[0], mol.boxl[1], mol.boxl[2], 90.0, 90.0, 90.0 ], sander.pme_input() )
        else:
            mm_inp = sander.pme_input()
            mm_inp.ifqnt = 1
            qm_inp = sander.QmInputOptions()
            qm_inp.qmmask = qm_mask
            qm_inp.qm_theory = qm_meth
            qm_inp.qmcharge = qm_chrg
            qm_inp.qmcut = qm_cut
            if( qm_meth.lower() == "extern" ):
                qm_inp.qm_ewald = 0
            self.obj = sander.setup( prmtop,
                mol.coor.ravel(), [ mol.boxl[0], mol.boxl[1], mol.boxl[2], 90.0, 90.0, 90.0 ], mm_inp, qm_inp )


    def stop( self ):
        if( sander.is_setup() ):
            sander.cleanup()


    def update_coor( self, mol: object ):
        sander.set_positions( mol.coor.ravel() )


    def get_func( self, mol: object ):
        self.update_coor( mol )
        ene = sander.energy_forces()[0].tot * qm3.data.K2J
        mol.func += ene
        return( ene )


    def get_grad( self, mol: object ):
        self.update_coor( mol )
        ene, grd = sander.energy_forces( True )
        ene = ene.tot * qm3.data.K2J
        grd = - grd.reshape( ( mol.natm, 3 ) ) * qm3.data.K2J
        mol.func += ene
        mol.grad += grd
        return( ene )
