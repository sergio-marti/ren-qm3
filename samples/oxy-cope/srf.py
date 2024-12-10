#!/usr/bin/env python3
#import  os
#os.environ["MPLBACKEND"] = "Agg"

import  numpy
import  qm3.utils.grids
import  matplotlib.pyplot as plt

g = qm3.utils.grids.grid()

g.regular( open( "pes.log" ), ( 31, 31 ), ( .15, .15 ) )
g.z = ( g.z - g.z.min() ) / 4.184
plt.clf()
plt.grid()
plt.title( "EQM [kcal/mol]" )
g.plot2d( levels = 40, fname = "pes.pdf" )
g.plot3d()

g.save( open( "pes.reg", "wt" ) )
