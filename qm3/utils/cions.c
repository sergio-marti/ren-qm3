#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


long        nat, npt[3];
double      *m_x, *m_y, *m_z, *m_q, *pot;
double      chrg, d_grd;
double      min_x, min_y, min_z;
char        *grd;
double      rx, ry, rz;


typedef struct { long ini, end; } p_range;


static void* molecular_potential( void *range ) {
    p_range     *c_rng = (p_range*) range;
    long        i, j, ix, iy, iz;
    double      cx, cy, cz;

    for( i = c_rng->ini; i < c_rng->end; i++ ) {
        iz = i;
        ix = iz / ( npt[1] * npt[2] );
        iz -= ix * npt[1] * npt[2];
        iy = iz / npt[2];
        iz -= iy * npt[2];
        cx = min_x + ((double) ix) * d_grd;
        cy = min_y + ((double) iy) * d_grd;
        cz = min_z + ((double) iz) * d_grd;
        for( j = 0; j < nat; j++ )
            pot[ix*npt[1]*npt[2]+iy*npt[2]+iz] += m_q[j] * chrg / sqrt( ( m_x[j] - cx ) * ( m_x[j] - cx ) + ( m_y[j] - cy ) * ( m_y[j] - cy ) + ( m_z[j] - cz ) * ( m_z[j] - cz ) );
    }
    return( NULL );
}


static void* charge_potential( void *range ) {
    p_range     *c_rng = (p_range*) range;
    long        i, ix, iy, iz;
    double      cx, cy, cz;

    for( i = c_rng->ini; i < c_rng->end; i++ ) {
        iz = i;
        ix = iz / ( npt[1] * npt[2] );
        iz -= ix * npt[1] * npt[2];
        iy = iz / npt[2];
        iz -= iy * npt[2];
        cx = min_x + ((double) ix) * d_grd;
        cy = min_y + ((double) iy) * d_grd;
        cz = min_z + ((double) iz) * d_grd;
        if( grd[ix*npt[1]*npt[2]+iy*npt[2]+iz] == 1 )
            pot[ix*npt[1]*npt[2]+iy*npt[2]+iz] += chrg * chrg / sqrt( ( rx - cx ) * ( rx - cx ) + ( ry - cy ) * ( ry - cy ) + ( rz - cz ) * ( rz - cz ) );
    }
    return( NULL );
}


static PyObject* cions( PyObject *self, PyObject *args ) {
    PyObject        *o_mole, *o_coor, *o_chrg;
    PyArrayObject   *m_coor, *m_chrg, *m_pnts;
    long            num, siz, cpu, *dim, xxx[2];
    long            i, j, k, ix, iy, iz, i_rad, I_rad, ii, jj, kk, l;
    double          d_ion, d_prt;
    double          rp, *ptr;
    time_t          t0;

    int             cp, ptje[11];

    pthread_t       *pid;
    p_range         *rng;


    if( PyArg_ParseTuple( args, "Olddddl", &o_mole, &num, &chrg, &d_grd, &d_ion, &d_prt, &cpu ) ) {
        t0 = time( NULL );
        // define current grid size
fprintf( stderr, "+ N.CPUs: %ld\n+ N.Ions: %ld\n+ Charge: %.1lf\n+ d_grd :%8.3lf\n+ r_ion :%8.3lf\n+ r_prt :%8.3lf\n", cpu, num, chrg, d_grd, d_ion, d_prt );

        o_coor = PyObject_GetAttrString( o_mole, "coor" );
        m_coor = (PyArrayObject*) PyArray_FROM_OT( o_coor, NPY_DOUBLE );
        o_chrg = PyObject_GetAttrString( o_mole, "chrg" );
        m_chrg = (PyArrayObject*) PyArray_FROM_OT( o_chrg, NPY_DOUBLE );

        dim = PyArray_SHAPE( m_chrg );
        nat = dim[0];
        m_x = malloc( nat * sizeof( double ) );
        m_y = malloc( nat * sizeof( double ) );
        m_z = malloc( nat * sizeof( double ) );
        m_q = malloc( nat * sizeof( double ) );
        ptr = (double*) PyArray_GETPTR2( m_coor, 0, 0 ); min_x = rx = m_x[0] = *ptr;
        ptr = (double*) PyArray_GETPTR2( m_coor, 0, 1 ); min_y = ry = m_y[0] = *ptr;
        ptr = (double*) PyArray_GETPTR2( m_coor, 0, 2 ); min_z = rz = m_z[0] = *ptr;
        ptr = (double*) PyArray_GETPTR1( m_chrg, 0 );                 m_q[0] = *ptr;
        for( i = 1; i < nat; i++ ) {
            ptr = (double*) PyArray_GETPTR2( m_coor, i, 0 ); m_x[i] = *ptr;
            min_x = min( min_x, m_x[i] ); rx = max( rx, m_x[i] );
            ptr = (double*) PyArray_GETPTR2( m_coor, i, 1 ); m_y[i] = *ptr;
            min_y = min( min_y, m_y[i] ); ry = max( ry, m_y[i] );
            ptr = (double*) PyArray_GETPTR2( m_coor, i, 2 ); m_z[i] = *ptr;
            min_z = min( min_z, m_z[i] ); rz = max( rz, m_z[i] );
            ptr = (double*) PyArray_GETPTR1( m_chrg, i );    m_q[i] = *ptr;
        }
        Py_DECREF( m_coor );
        Py_DECREF( m_chrg );
        Py_DECREF( o_coor );
        Py_DECREF( o_chrg );
        min_x -= max( d_ion, d_prt ); min_y -= max( d_ion, d_prt ); min_z -= max( d_ion, d_prt );
        rx += max( d_ion, d_prt ); ry += max( d_ion, d_prt ); rz += max( d_ion, d_prt );
        npt[0] = (long)( ( rx - min_x ) / d_grd ) + 1;
        npt[1] = (long)( ( ry - min_y ) / d_grd ) + 1;
        npt[2] = (long)( ( rz - min_z ) / d_grd ) + 1;
        siz    = npt[0] * npt[1] * npt[2];
fprintf( stderr, "+ min_X :%8.3lf\n+ min_Y :%8.3lf\n+ min_Z :%8.3lf\n",min_x,min_y,min_z);
fprintf( stderr, "+ Grid  : Nx (%ld) * Ny (%ld) * Nz (%ld) => %ld points\n",npt[0], npt[1], npt[2], siz );

        // allocate grid and initialize
        grd = malloc( siz * sizeof( char ) );
        pot = malloc( siz * sizeof( double ) );
        for( l = 0, i = 0; i < npt[0]; i++ )
            for( j = 0; j < npt[1]; j++ )
                for( k = 0; k < npt[2]; k++ ) {
                    grd[l] = 1; pot[l] = .0; l++;
                }

fprintf( stderr, "+ Grid  : pruning molecule... " );
        // prune molecule
        i_rad = (long) ( d_prt / d_grd );
        I_rad = i_rad * i_rad;
        for( i = 0; i < nat; i++ ) {
            ix = (long) ( ( m_x[i] - min_x ) / d_grd );
            iy = (long) ( ( m_y[i] - min_y ) / d_grd );
            iz = (long) ( ( m_z[i] - min_z ) / d_grd );
            for( ii = - i_rad; ii <= i_rad; ii++ )
                for( jj = - i_rad; jj <= i_rad; jj++ )
                    for( kk = - i_rad; kk <= i_rad; kk++ )
                        if( ii*ii+jj*jj+kk*kk <= I_rad &&
                            ix + ii < npt[0] && ix + ii >= 0 &&
                            iy + jj < npt[1] && iy + jj >= 0 &&
                            iz + kk < npt[2] && iz + kk >= 0  )
                                grd[(ix+ii)*npt[1]*npt[2]+(iy+jj)*npt[2]+(iz+kk)] = 0;
        }
fprintf( stderr, "done!\n" );

        // compute molecular "potential" on remaining grid points
fprintf( stderr, "+ Grid  : computing molecular potential on remaining grid points... " );
        pid = malloc( cpu * sizeof( *pid ) );
        rng = malloc( cpu * sizeof( *rng ) );
        i = siz / cpu; 
        for( j = 0; j < cpu; j++ ) { rng[j].ini = j*i; rng[j].end = (j+1)*i; }
        rng[cpu-1].end += siz % cpu;
        for( i = 0; i < cpu; i++ ) pthread_create( &pid[i], NULL, molecular_potential, (void*) &rng[i] );
        for( i = 0; i < cpu; i++ ) pthread_join( pid[i], NULL );
        free( pid ); free( rng );
fprintf( stderr, "done!\n" );

        // clean memory: molecular data
        free( m_x ); free( m_y ); free( m_z ); free( m_q );

        // start placing ions...
        xxx[0] = num; xxx[1] = 3;
        m_pnts = (PyArrayObject*) PyArray_ZEROS( 2, xxx, NPY_DOUBLE, 0 );
for( i = 0; i < 11; i++ ) ptje[i] = 0;
        i_rad = (long) ( d_ion / d_grd );
        I_rad = i_rad * i_rad;
fprintf( stderr, "+ Search: " );
        for( l = 0; l < num; l++ ) {
cp = ((int)(10.*((float)l)/((float)num)));
if( ptje[cp] == 0 ) { ptje[cp] = 1; fprintf( stderr, "%d%%...", cp*10 ); }
            // search for the preferred point in grid
            ix = 0; iy = 0; iz = 0; rp = 1.E+308;
            for( i = 0; i < npt[0]; i++ )
                for( j = 0; j < npt[1]; j++ )
                    for( k = 0; k < npt[2]; k++ )
                        if( grd[i*npt[1]*npt[2]+j*npt[2]+k] == 1 )
                            if( pot[i*npt[1]*npt[2]+j*npt[2]+k] < rp ) {
                                ix = i; iy = j; iz = k; rp = pot[i*npt[1]*npt[2]+j*npt[2]+k];
                            }
            ptr = (double*) PyArray_GETPTR2( m_pnts, l, 0 );
            *ptr = min_x + ((double) ix) * d_grd;
            ptr = (double*) PyArray_GETPTR2( m_pnts, l, 1 );
            *ptr = min_y + ((double) iy) * d_grd;
            ptr = (double*) PyArray_GETPTR2( m_pnts, l, 2 );
            *ptr = min_z + ((double) iz) * d_grd;
            if( l < num - 1 ) { // skip this for the last ion...
                // prune around current point
                for( ii = - i_rad; ii <= i_rad; ii++ )
                    for( jj = - i_rad; jj <= i_rad; jj++ )
                        for( kk = - i_rad; kk <= i_rad; kk++ )
                            if( ii*ii+jj*jj+kk*kk <= I_rad &&
                                ix + ii < npt[0] && ix + ii >= 0 &&
                                iy + jj < npt[1] && iy + jj >= 0 &&
                                iz + kk < npt[2] && iz + kk >= 0  )
                                    grd[(ix+ii)*npt[1]*npt[2]+(iy+jj)*npt[2]+(iz+kk)] = 0;
                // recalculate potential for current point
                pid = malloc( cpu * sizeof( *pid ) );
                rng = malloc( cpu * sizeof( *rng ) );
                i = siz / cpu; 
                for( j = 0; j < cpu; j++ ) { rng[j].ini = j*i; rng[j].end = (j+1)*i; }
                rng[cpu-1].end += siz % cpu;
                for( i = 0; i < cpu; i++ ) pthread_create( &pid[i], NULL, charge_potential, (void*) &rng[i] );
                for( i = 0; i < cpu; i++ ) pthread_join( pid[i], NULL );
                free( pid ); free( rng );
            }
    }
fprintf( stderr, "100%%\n" );

        // clean memory: grid data
        free( grd ); free( pot );

fprintf( stderr, "+ Time  : %ld sec\n", time( NULL ) - t0 );
        return( (PyObject*) m_pnts );
    } else {
        Py_INCREF( Py_None ); return( Py_None );
    }
}


static struct PyMethodDef methods [] = {
    { "counter_ions", (PyCFunction)cions, METH_VARARGS },
    { 0, 0, 0 }
};


static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_cions",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__cions( void ) {
    PyObject    *my_module;
    my_module = PyModule_Create( &moddef );
    import_array();
    return( my_module );
}
