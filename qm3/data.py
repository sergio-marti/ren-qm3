import  numpy

#
#    http://physics.nist.gov/cuu/Constants/index.html
#

C   = 299792458.0      # m.s-1
NA  = 6.02214076e23    # mol-1
H   = 6.62607015e-34   # J.s
A0  = 0.52917721093    # 1e-10.m
KB  = 1.380649e-23     # J.K-1
R   = 8.3144626        # J.mol-1.K-1
ME  = 9.109387015e-31  # kg
EV  = 1.602176634e-19  # eV >> J
HA  = 4.3597447222e-18 # Ha >> J

K2J = 4.184            # kcal >> kJ
J2K = 0.239005736138   # kJ >> kcal
H2K = 627.509474062034 # Ha >> kcal.mol-1
H2J = 2625.49963947555 # Ha >> kJ.mol-1
R2D = 180.0/numpy.pi   # Radians to dregrees


MXLAT = 1.0e300


# g/mol
mass = numpy.array( [ 0.00, 1.00794, 4.00260, 6.94100, 9.01218, 10.8110, 12.0107, 14.0067, 15.9994, 18.9984, 20.1797, 22.9898,
    24.3050, 26.9815, 28.0855, 30.9738, 32.0650, 35.4530, 39.9480, 39.0983, 40.0780, 44.9559, 47.8670, 50.9415, 51.9961,
    54.9380, 55.8450, 58.9332, 58.6934, 63.5460, 65.3900, 69.7230, 72.6400, 74.9216, 78.9600, 79.9040, 83.8000, 85.4678,
    87.6200, 88.9059, 91.2240, 92.9064, 95.9400, 98.9063, 101.0700, 102.9060, 106.4200, 107.8680, 112.4110, 114.8180,
    118.7100, 121.7600, 127.6000, 126.9040, 131.2930, 132.9050, 137.2370, 138.9050, 140.1160, 140.9080, 144.2400, 146.9150,
    150.3600, 151.9640, 157.2500, 158.9250, 162.5000, 164.9300, 167.2590, 168.9340, 173.0400, 174.9670, 178.4900, 180.9480,
    183.8400, 186.2070, 190.2300, 192.2170, 195.0780, 196.9670, 200.5900, 204.3830, 207.2000, 208.9800, 208.9820, 209.9870,
    222.0180, 223.0200, 226.0250, 227.0280, 232.0380, 231.0360, 238.0290, 237.0480, 244.0640, 243.0610, 247.0700, 247.0700,
    251.0800, 252.0830, 257.0950, 258.0990, 259.1010, 262.1100, 261.1090, 262.1140, 263.1190, 262.1230, 265.1310, 266.1380 ],
    dtype=numpy.float64 )


symbol = numpy.array( [ "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt" ], dtype=numpy.unicode )


rsymbol = {}
for i,j in enumerate( symbol ):
    rsymbol[j] = i


def water_density( temp = 300.0 ):
    tmp = min( max( -8.0, temp - 273.15 ), 108.0 )
    tab = [ 0.998120,0.998398,0.998650,0.998877,0.999080,0.999259,0.999417,0.999553,0.999669,0.999765,
        0.9998425,0.9999015,0.9999429,0.9999672,0.9999750,0.9999668,0.9999432,0.9999045,0.9998512,0.9997838,
        0.9997026,0.9996018,0.9995004,0.9993801,0.9992474,0.9991026,0.9989460,0.9987779,0.9985986,0.9984082,
        0.9982071,0.9979955,0.9977735,0.9975415,0.9972995,0.9970479,0.9967867,0.9965162,0.9962365,0.9959478,
        0.9956502,0.9953440,0.9950292,0.9947060,0.9943745,0.9940349,0.9936872,0.9933316,0.9929683,0.9925973,
        0.9922187,0.9918327,0.9914394,0.9910388,0.9906310,0.9902162,0.9897944,0.9893657,0.9889303,0.9884881,
        0.9880393,0.9875839,0.9871220,0.9866537,0.9861791,0.9856982,0.9852111,0.9847178,0.9842185,0.9837132,
        0.9832018,0.9826846,0.9821615,0.9816327,0.9810981,0.9805578,0.9800118,0.9794603,0.9789032,0.9783406,
        0.9777726,0.9771991,0.9766203,0.9760361,0.9754466,0.9748519,0.9742520,0.9736468,0.9730366,0.9724212,
        0.9718007,0.9711752,0.9705446,0.9699091,0.9692686,0.9686232,0.9679729,0.9673177,0.9666576,0.9659927,
        0.9653230,0.9646486,0.9639693,0.9632854,0.9625967,0.9619033,0.9612052,0.9605025,0.9597951,0.9590831,
        0.9583665,0.957662,0.956937,0.956207,0.955472,0.954733,0.953989,0.953240,0.952488,0.941730 ]
    i = numpy.floor( tmp )
    p = tmp - i
    p2m1 = p * p - 1.0
    p2m4 = p2m1 - 3.0
    i = int( i ) + 10
    d = p2m1 * p * ( p - 2.0 ) * tab[i-2] / 24.0 - ( p - 1.0 ) * p * p2m4 * tab[i-1] / 6.0 + p2m1 *p2m4 * tab[i] / 4.0 - ( p + 1.0 ) * p * p2m4 * tab[i+1] / 6.0 + p2m1 * p * ( p + 2.0 ) * tab[i+2] / 24.0
    return( d )

