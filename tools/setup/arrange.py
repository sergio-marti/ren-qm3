#!/usr/bin/env python3
import  qm3
import  qm3.data
import  qm3.utils._conn
import  numpy
import  networkx

# reassign atom labels of docked substrate in the PDB
# using as template the amber file... (all atoms)

m = qm3.molecule()
m.pdb_read( open( "amber.ac" ) )
m.guess_atomic_numbers()
m.nx = networkx.Graph()
for i in range( m.natm ):
    m.nx.add_node( i, element = qm3.data.symbol[m.anum[i]] )
m.nx.add_edges_from( qm3.utils._conn.connectivity( 4, m.anum, m.coor ) )

o = qm3.molecule()
o.pdb_read( open( "docked.pdb" ) )
o.guess_atomic_numbers()
o.nx = networkx.Graph()
for i in range( o.natm ):
    o.nx.add_node( i, element = qm3.data.symbol[o.anum[i]] )
o.nx.add_edges_from( qm3.utils._conn.connectivity( 4, o.anum, o.coor ) )

node_match = lambda n1, n2: n1['element'] == n2['element']
GM = networkx.algorithms.isomorphism.GraphMatcher( m.nx, o.nx, node_match = node_match )
if( GM.subgraph_is_isomorphic() ):
    for im, io in GM.mapping.items():
        print( io, im, o.labl[io], m.labl[im] )
        o.labl[io] = m.labl[im]
    o.pdb_write( open( "docked-amber.pdb", "wt" ) )
