import  numpy
import  qm3
import  qm3.engines.molmech
import  io

f = io.StringIO( """21

C           0.0030254337        0.1368072224       -0.1380533098
C           0.9421625577        0.6221999644        0.7832233512
C           1.7829740380       -0.3344795901        1.6055713643
C           2.7779290585        0.4626409463        2.4964835458
C           2.5859266731       -1.2589070728        0.7120083269
N           0.8462250220       -1.1375072277        2.3800827665
H           3.4132916814       -0.2313047712        3.0497889027
C           2.0606591612        1.3308795391        3.5380329010
O           3.6019862867        1.2257907656        1.6316816060
H          -0.1067126352       -0.9266299701       -0.2930752312
H          -0.5889930257        0.8282181054       -0.7208246136
H           1.0762115351        1.6962100445        0.9172732643
H           4.0101127610        0.6137743988        1.0062502959
O           1.5713752919        2.4407576646        3.2353507488
O           1.9475154496        0.8848058385        4.6989251121
H           1.0181384599       -0.9910870323        3.3715617569
H           0.9302803048       -2.1177255456        2.1633482574
H          -0.0918155196       -0.8238185811        2.1742513311
N           3.1253511449       -0.7995303875       -0.4595118192
H           2.8740994691       -2.2453281239        1.0563989277
H           2.8897268521        0.1470538127       -0.7390574846
""" )

mol = qm3.molecule()
mol.xyz_read( f )
mol.guess_atomic_numbers()
mol.chrg[5]  = +1.0
mol.chrg[13] = -0.5
mol.chrg[14] = -0.5

eng = qm3.engines.molmech.run( mol )
eng.initialize( mol, impr = [ [ 7, 13, 14, 3, 300.0, 0.0 ] ] )
eng.load_parameters( mol )
eng.cut_on   = -1
eng.cut_off  = -1
eng.cut_list = -1

print( 80*"=" )
print( "%-4s%-4s%8s%10s "%( "", "Atm", "Chrg", "Typ" ), "Con" )
for i in range( mol.natm ):
    print( "%-4d%-4s%8.3lf%10s "%( i, mol.labl[i], mol.chrg[i], mol.type[i] ), eng.conn[i] )
print( 80*"-" )
print( "Charge: %8.3lf"%( sum( mol.chrg ) ) )
print( 80*"-" )

mol.func = 0.0
mol.grad = numpy.zeros( ( mol.natm, 3 ) )
print()
eng.get_grad( mol, qprint = True )
print()
print( mol.grad )

print( round( mol.func, 1 ), "/ -218.2" )
print( round( numpy.linalg.norm( mol.grad ), 1 ), "/ 11.6" )
