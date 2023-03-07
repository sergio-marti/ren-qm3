import  numpy
import  qm3
import  qm3.engines.mopac
import  qm3.actions.minimize
import  qm3.actions.neb
import  io


mol = qm3.molecule()
f = io.StringIO( """26

C           3.3785117988       -1.5597014566       -2.4704385269
C           3.1122692398       -2.9582320727       -2.8139748305
C           3.8210019909       -3.5780686151       -3.7692285658
C           5.0002519742       -2.9436033696       -4.4419416737
O           4.8704916160       -2.9527729687       -5.8605685753
C           5.2049404639       -1.4943852765       -4.0506684093
C           4.2051596452       -0.8188539011       -3.3327152949
O           6.4035739183       -1.6742746366       -2.6551504734
C           5.8346999982       -2.2686267776       -1.6581345136
C           4.7958304830       -1.6759081138       -0.9210660452
C           2.3706340619       -0.8296317728       -1.6726100872
O           2.3779435694        0.3615329675       -1.3430730790
O           1.3147857008       -1.5752021892       -1.2385244766
C           6.1855655523       -3.6833497553       -1.3475102338
O           5.8761258005       -4.3357269815       -0.3450306891
O           6.9721074595       -4.3093515800       -2.2659538447
H           5.9299913978       -3.5283440088       -4.1529613348
H           4.7255738475       -3.8707805187       -6.1197016355
H           5.9016033543       -0.9315982480       -4.6978623557
H           4.7781845855       -0.5804263514       -0.7947993767
H           4.3007596704       -2.2641328512       -0.1331495493
H           2.2855076360       -3.4534270262       -2.2783875100
H           3.5836246443       -4.6030885517       -4.0909671045
H           4.2069725062        0.2816471597       -3.2777512137
H           0.7132224994       -1.0100977893       -0.7239810064
H           7.2054065864       -5.1951553158       -1.9398695949
""" )
mol.xyz_read( f )
mol.guess_atomic_numbers()
mol.engines["qm"] = qm3.engines.mopac.run( mol, "AM1", 0, 1 )
tran = mol.coor.copy()

f = io.StringIO( """26

C           3.0402546370       -1.5190627524       -2.7540187775
C           2.8278619310       -2.9270630973       -3.0438408278
C           3.7001959760       -3.6084408361       -3.8012713258
C           4.9561542828       -3.0065462998       -4.3450774500
O           4.8716479314       -2.8007649012       -5.7521645192
C           5.3142903105       -1.6290832892       -3.7505749590
C           4.1652364017       -0.8879082543       -3.1542748860
O           6.3676879756       -1.7743230006       -2.7741336185
C           5.9828813328       -2.3216193076       -1.5578138065
C           5.3627974135       -1.6082885500       -0.6045448514
C           2.0412306125       -0.7822985017       -1.9658865081
O           2.0334040611        0.4205984539       -1.6780466469
O           1.0372967566       -1.5389277643       -1.4372187656
C           6.3946173754       -3.7175084724       -1.3166279158
O           6.1764309735       -4.4013066036       -0.3098639829
O           7.0860459750       -4.3158124360       -2.3255291288
H           5.8212920833       -3.7070519844       -4.1302083226
H           4.6002857756       -3.6400374640       -6.1426317036
H           5.8200148777       -1.0108748001       -4.5525035428
H           5.0461186161       -0.5682008890       -0.7654729598
H           5.1350019292       -2.0492296390        0.3769919021
H           1.9257118953       -3.3962486123       -2.6183339604
H           3.5380514319       -4.6674420723       -4.0532320514
H           4.3073218711        0.1961140585       -3.0053871356
H           0.4716601170       -0.9743562216       -0.8834817962
H           7.3312474567       -5.2158767640       -2.0508724605
""" )
mol.xyz_read( f )
reac = mol.coor.copy()

f = io.StringIO( """26

C           3.6786748910       -1.6107856718       -2.2384838153
C           3.2479211729       -2.9588681167       -2.7365535069
C           3.7771696468       -3.5132455894       -3.8308058966
C           4.8864805613       -2.8500497835       -4.5886353937
O           4.8431896654       -3.1235588475       -5.9827523184
C           4.8387046819       -1.3614194231       -4.4225268213
C           4.2973306642       -0.7976144604       -3.3398674153
O           6.9320427061       -1.5384507349       -1.8406791831
C           6.0145338995       -2.2468296348       -1.4383332370
C           4.6547870316       -1.7442624444       -1.0548024096
C           2.4869458769       -0.8393656295       -1.6891878759
O           2.3714177969        0.3814068120       -1.5576902211
O           1.4456398857       -1.6030099689       -1.2584472487
C           6.2325951358       -3.7212943325       -1.1991976862
O           5.8582044319       -4.3921533433       -0.2360114920
O           6.9323314389       -4.3477334950       -2.1787426561
H           5.8733346963       -3.2324145587       -4.1716388145
H           4.7477260527       -4.0785027387       -6.0781834215
H           5.2816598139       -0.7758544780       -5.2418058371
H           4.7881809891       -0.7245620103       -0.5967646836
H           4.2040539032       -2.4318323261       -0.2889975957
H           2.4485334025       -3.4491409008       -2.1576049357
H           3.4284066950       -4.4823360984       -4.2167152031
H           4.2673630755        0.2939605475       -3.1914133048
H           0.7364095795       -1.0261949255       -0.9268183021
H           7.0811023059       -5.2774478484       -1.9333607252
""" )
mol.xyz_read( f )
prod = mol.coor.copy()

gues = qm3.actions.neb.distribute( 20, [ reac, tran, prod ] )

with open( "chain", "wt" ) as f:
    for g in gues:
        mol.coor = g
        mol.xyz_write( f )


obj = qm3.actions.neb.serial( mol, gues, 100 )

qm3.actions.minimize.fire( obj, print_frequency = 1, gradient_tolerance = len( gues ) * 0.1 )
