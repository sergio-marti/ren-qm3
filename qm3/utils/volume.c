#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


/*
double r_vdw[110] = { 1.10, 1.10, 1.40, 1.81, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27,
        1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31, 2.30, 2.15, 2.05, 2.05, 2.05,
        2.05, 2.00, 2.00, 2.00, 2.10, 1.87, 2.11, 1.85, 1.90, 1.83, 2.02, 3.03, 2.49, 2.40,
        2.30, 2.15, 2.10, 2.05, 2.05, 2.00, 2.05, 2.10, 2.20, 2.20, 1.93, 2.17, 2.06, 1.98,
        2.16, 3.43, 2.68, 2.50, 2.48, 2.47, 2.45, 2.43, 2.42, 2.40, 2.38, 2.37, 2.35, 2.33,
        2.32, 2.30, 2.28, 2.27, 2.25, 2.20, 2.10, 2.05, 2.00, 2.00, 2.05, 2.10, 2.05, 1.96,
        2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.00, 2.40, 2.00, 2.30, 2.00, 2.00, 2.00,
        2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00 };
*/


typedef struct ptn_node { long i, j, k; struct ptn_node *n; } ptn_lst;
typedef struct sph_node { long r; ptn_lst *p; struct sph_node *n; } sph_lst;
typedef struct { long _i0, _if, *npt; char *grd; double dsp, *bmn, *crd; sph_lst **hsh; } targ;

typedef struct srf_node { double x, y, z; struct srf_node *n; } srf_lst;


double* fibonacci_sphere( long dime, long *size ) {
    long    i, fib_a = 0, fib_b = 1, swp;
    double  *out = NULL, c1, c2, tt, ff;

    for( i = 0; i < dime; i++ ) {
        swp = fib_b;
        fib_b = fib_a + fib_b;
        fib_a = swp;
    }
    *size = fib_b;
    out   = (double*) malloc( 3 * fib_b * sizeof( double ) );
    c1    = 2.0 / fib_b;
    c2    = 2.0 * M_PI * fib_b / fib_a;
    for( i = 0; i < fib_b; i++ ) {
        tt = acos( 1.0 - c1 * i );
        ff = i * c2;
        out[3*i]   = sin( tt ) * cos( ff );
        out[3*i+1] = sin( tt ) * sin( ff );
        out[3*i+2] = cos( tt );
    }
    return( out );
}


static PyObject* __molecular_surf( PyObject *self, PyObject *args ){
    PyObject        *o_xyz, *o_rad;
    PyArrayObject   *m_xyz, *m_rad;
    long            i, j, k, l, nit = 0, *flg, i3, j3;
    long            npt = 14, siz = 0, *dim;
    double          *sph, *ptr_f, *rad, *crd, ri2, rj2, ds2, rr2;
    srf_lst         *srf = NULL, *cur;

    if( PyArg_ParseTuple( args, "OO|l", &o_rad, &o_xyz, &npt ) ) {
        m_rad = (PyArrayObject*) PyArray_FROM_OT( o_rad, NPY_DOUBLE );
        m_xyz = (PyArrayObject*) PyArray_FROM_OT( o_xyz, NPY_DOUBLE );
        dim   = PyArray_SHAPE( m_xyz );
        rad   = (double*) malloc(     dim[0] * sizeof( double ) );
        crd   = (double*) malloc( 3 * dim[0] * sizeof( double ) );
        for( i = 0; i < dim[0]; i++ ) {
            ptr_f = (double*) PyArray_GETPTR1( m_rad, i );
            rad[i] = *ptr_f;
            for( j = 0; j < 3; j++ ) {
                ptr_f = (double*) PyArray_GETPTR2( m_xyz, i, j );
                crd[3*i+j] = *ptr_f;
            }
        }
        Py_DECREF( m_rad );
        Py_DECREF( m_xyz );

        srf    = (srf_lst*)  malloc( sizeof( srf_lst ) );
        srf->n = NULL;
        cur    = srf;

//        sph = fibonacci_sphere( npt, &siz );
//        flg = (long*) malloc( siz * sizeof( long ) );

        for( i = 0; i < dim[0]; i++ ) {
            i3  = i * 3;
            ri2 = rad[i] * rad[i];
            //for( k = 0; k < siz; k++ ) flg[k] = 0;
            // ----------------------------------------------------------------------
            sph = fibonacci_sphere( max( npt, lround( rad[i] * 10.0 ) ), &siz );
            flg = (long*) calloc( siz, sizeof( long ) );
            // ----------------------------------------------------------------------
            for( j = 0; j < dim[0]; j++ ) {
                if( i != j ) {
                    j3  = j * 3;
                    rj2 = rad[j] * rad[j];
                    rr2 = ri2 + rj2 + 2.0 * rad[i] * rad[j];
                    for( ds2 = 0.0, l = 0; l < 3; l++ )
                        ds2 += ( crd[i3+l] - crd[j3+l] ) * ( crd[i3+l] - crd[j3+l] );
                    if( ds2 <= rr2 ) {
                        for( k = 0; k < siz; k++ ) {
                            for( ds2 = 0.0, l = 0; l < 3; l++ )
                                ds2 += ( rad[i] * sph[k*3+l] + crd[i3+l] - crd[j3+l] ) * ( rad[i] * sph[k*3+l] + crd[i3+l] - crd[j3+l] );
                            flg[k] += ( ds2 <= rj2 );
                        }
                    }
                }
            }
            for( k = 0; k < siz; k++ ) {
                if( flg[k] == 0 ) {
                    nit++;
                    cur->n = (srf_lst*)  malloc( sizeof( srf_lst ) );
                    cur->n->x = rad[i] * sph[k*3]   + crd[i3];
                    cur->n->y = rad[i] * sph[k*3+1] + crd[i3+1];
                    cur->n->z = rad[i] * sph[k*3+2] + crd[i3+2];
                    cur->n->n = NULL;
                    cur = cur->n;
                }
            }
            free( sph ); free( flg );
        }

        dim[0] = nit; dim[1]  = 3;
        m_xyz  = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
        cur = srf->n; free( srf ); srf = cur;
        for( i = 0; i < nit; i++ ) {
            ptr_f = (double*) PyArray_GETPTR2( m_xyz, i, 0 );
            *ptr_f = cur->x;
            ptr_f = (double*) PyArray_GETPTR2( m_xyz, i, 1 );
            *ptr_f = cur->y;
            ptr_f = (double*) PyArray_GETPTR2( m_xyz, i, 2 );
            *ptr_f = cur->z;
            cur = cur->n; free( srf ); srf = cur;
        }
        free( crd ); free( rad ); //free( flg ); free( sph );

        return( (PyObject*) m_xyz );
    }
    Py_INCREF( Py_None ); return( Py_None );
}



void* __fill_atoms( void *args ) {
    targ    *a = (targ*) args;
    long    l, l3, i, j, k, n01;
    ptn_lst *c;

    n01 = a->npt[0] * a->npt[1];
    for( l = a->_i0; l < a->_if; l++ ) {
        l3 = l * 3;
        i  = (long)( ( a->crd[l3]   - a->bmn[0] ) / a->dsp );
        j  = (long)( ( a->crd[l3+1] - a->bmn[1] ) / a->dsp );
        k  = (long)( ( a->crd[l3+2] - a->bmn[2] ) / a->dsp );
        c  = a->hsh[l]->p;
        while( c != NULL ) {
            a->grd[ (i + c->i) + (j + c->j) * a->npt[0] + (k + c->k) * n01 ] += 1;
            c = c->n;
        }
    }
    return( NULL );
}


static PyObject* __molecular_grid( PyObject *self, PyObject *args ){
    PyObject        *o_xyz, *o_rad, *o_pdb;
    PyArrayObject   *m_xyz, *m_rad;
    long            i, j, k, l, l3, cpu, *siz, rad, rd2, *rng, nit, knd = 0;
    double          mad, *ptr_f;
    double          vol = 0.0;
    double          bmax[3] = { -9999., -9999., -9999. };
    double          bmin[3] = { +9999., +9999., +9999. };
    long            npt[3];
    char            *grd;
    double          *crd, dsp, d3;
    sph_lst         *sph, **hsh;
    sph_lst         *sph_cur;
    ptn_lst         *ptn_cur;
    pthread_t       *pid;
    targ            *arg;
    time_t          t0;

    dsp = 0.05;
    o_pdb = Py_False;
    if( PyArg_ParseTuple( args, "lOO|dO", &cpu, &o_rad, &o_xyz, &dsp, &o_pdb ) ) {

        t0    = time( NULL );
        d3    = dsp * dsp * dsp;
        m_rad = (PyArrayObject*) PyArray_FROM_OT( o_rad, NPY_DOUBLE );
        m_xyz = (PyArrayObject*) PyArray_FROM_OT( o_xyz, NPY_DOUBLE );
        siz   = PyArray_SHAPE( m_rad );
        crd   = (double*) malloc( 3 * siz[0] * sizeof( double ) );
        if( siz[0] < cpu ) { cpu = 1; }
fprintf(stderr,"CPU: %ld\n",cpu);
fprintf(stderr,"DSP: %lf\n",dsp);

        hsh = (sph_lst**) malloc( siz[0] * sizeof( sph_lst* ) );
        sph = (sph_lst*)  malloc( sizeof( sph_lst ) );
        sph->r = -1;
        sph->p = NULL;
        sph->n = NULL;
        mad = .0;
        for( l = 0; l < siz[0]; l++ ) {
            ptr_f = (double*) PyArray_GETPTR1( m_rad, l );
            mad = max( mad, *ptr_f );
            rad = (long)( *ptr_f / dsp );

            sph_cur = sph;
            while( sph_cur != NULL && sph_cur->r != rad ) { sph_cur = sph_cur->n; }
            if( sph_cur == NULL ) {
                for( sph_cur = sph; sph_cur->n != NULL; sph_cur = sph_cur->n );

                sph_cur->n = (sph_lst*) malloc( sizeof( sph_lst ) );
                sph_cur->n->r = rad;
                sph_cur->n->n = NULL;
                sph_cur->n->p = (ptn_lst*) malloc( sizeof( ptn_lst ) );
                sph_cur->n->p->i = 9999;
                sph_cur->n->p->j = 9999;
                sph_cur->n->p->k = 9999;
                sph_cur->n->p->n = NULL;

                ptn_cur = sph_cur->n->p;
                rd2 = rad * rad;
                for( i = -rad; i <= rad; i++ ) {
                    for( j = -rad; j <= rad; j++ ) {
                        for( k = -rad; k <= rad; k++ ) {
                            if( i*i + j*j + k*k <= rd2 ) {
                                ptn_cur->n = (ptn_lst*) malloc( sizeof( ptn_lst ) );
                                ptn_cur->n->i = i;
                                ptn_cur->n->j = j;
                                ptn_cur->n->k = k;
                                ptn_cur->n->n = NULL;
                                ptn_cur = ptn_cur->n;    
                            }
                        }
                    }
                }
                ptn_cur = sph_cur->n->p;
                sph_cur->n->p = sph_cur->n->p->n;
                free( ptn_cur );
                sph_cur = sph_cur->n;
                knd++;
            }
            hsh[l] = sph_cur;
            l3     = l * 3;
            for( i = 0; i < 3; i++ ) {
                ptr_f = (double*) PyArray_GETPTR2( m_xyz, l, i );
                crd[l3+i] = *ptr_f;
                bmin[i] = min( bmin[i], crd[l3+i] );
                bmax[i] = max( bmax[i], crd[l3+i] );
            }
        }
        Py_DECREF( m_rad );
        Py_DECREF( m_xyz );
        mad *= 1.1;
        for( i = 0; i < 3; i++ ) { 
            bmin[i] -= mad; bmax[i] += mad;
            npt[i] = (long)( ( bmax[i] - bmin[i] ) / dsp ) + 1;
        }

fprintf(stderr,"MIN: %8.3lf%8.3lf%8.3lf\n",bmin[0],bmin[1],bmin[2]);
fprintf(stderr,"MAX: %8.3lf%8.3lf%8.3lf\n",bmax[0],bmax[1],bmax[2]);
fprintf(stderr,"NPT: %8ld%8ld%8ld\n",npt[0],npt[1],npt[2]);
fprintf(stderr,"KND: %8ld\n",knd);

        grd = (char*) malloc( npt[0] * npt[1] * npt[2] * sizeof( char ) );
        for( l = 0; l < npt[0] * npt[1] * npt[2]; l++ ) grd[l] = 0;

        pid = (pthread_t*) malloc( cpu * sizeof( pthread_t ) );
        arg = (targ*) malloc( cpu * sizeof( targ ) );
        rng = (long*) malloc( ( cpu + 1 ) * sizeof( long ) );
        for( l = 0; l <= cpu; l++ ) rng[l] = 0;

        nit = (long)round( (float)siz[0] / (float)cpu );
        if( nit != 0 ) for( l = 0; l < cpu; l++ ) rng[l] = l * nit;
        rng[cpu] = siz[0];
        for( l = 0; l < cpu; l++ ) {
            arg[l]._i0 = rng[l];
            arg[l]._if = rng[l+1];
            arg[l].dsp = dsp;
            arg[l].npt = npt;
            arg[l].bmn = bmin;
            arg[l].crd = crd;
            arg[l].grd = grd;
            arg[l].hsh = hsh;
            pthread_create( &pid[l], NULL, __fill_atoms, (void*) &arg[l] );
        }
        for( l = 0; l < cpu; l++ ) pthread_join( pid[l], NULL );

        if( o_pdb == Py_True ) {
            long    n01 = npt[0] * npt[1];
            FILE*     fd = fopen( "volume.pdb", "wt" );
            for( i = 1; i < npt[0] - 1; i++ )
                for( j = 1; j < npt[1] - 1; j++ )
                    for( k = 1; k < npt[2] - 1; k++ )
                        if( grd[i + j * npt[0] + k * n01 ] > 0 ) { 
                            vol += 1.0;
                            if( ! ( grd[i-1 + j * npt[0] + k * n01] > 0 && 
                                    grd[i+1 + j * npt[0] + k * n01] > 0 && 
                                    grd[i + (j-1) * npt[0] + k * n01] > 0 && 
                                    grd[i + (j+1) * npt[0] + k * n01] > 0 && 
                                    grd[i + j * npt[0] + (k-1) * n01] > 0 && 
                                    grd[i + j * npt[0] + (k+1) * n01] > 0 ) )
                                fprintf( fd, "ATOM      1  H   SRF     1    %8.3lf%8.3lf%8.3lf  1.00  0.00    SRF\n",
                                    bmin[0] + i * dsp, bmin[1] + j * dsp, bmin[2] + k * dsp );
                        }
        } else {
            for( l = 0; l < npt[0] * npt[1] * npt[2]; l++ ) {
                if( grd[l] > 0 ) { vol += 1.0; }
            }
        }
        vol *= d3;
fprintf(stderr,"VOL: %lf _A^3\n",vol);

        free( pid ); free( rng ); free( arg );
        free( crd ); 
        sph_cur = sph;
        while( sph_cur != NULL ) {
            ptn_cur = sph_cur->p;
            while( ptn_cur != NULL ) {
                ptn_cur = ptn_cur->n;
                free( sph_cur->p );
                sph_cur->p = ptn_cur;
            }
            sph_cur = sph_cur->n;
            free( sph );
            sph = sph_cur;
        }
        free( hsh );
        free( grd );

fprintf( stderr, "TIM: %ld _sec\n", time( NULL ) - t0 );
        return( Py_BuildValue( "d", vol ) );
    }
    Py_INCREF( Py_None ); return( Py_None );
}



long __collide( double x, double y, double z, double prb, double dsp, long siz, double *rad, double *crd ) {
    long    i, f = 0;
    double  dx, dy, dz, rr;

    for( i = 0; i < siz && f == 0; i++ ) {
        dx = x - crd[i*3];
        dy = y - crd[i*3+1];
        dz = z - crd[i*3+2];
        rr = rad[i] + prb - dsp;
        f  = ( dx * dx + dy * dy + dz * dz ) <= ( rr * rr );
    }
    return( f );
}


static PyObject* __cavity_grid( PyObject *self, PyObject *args ){
    PyObject        *o_xyz, *o_rad, *o_cen, *o_pdb, *o_trn;
    PyArrayObject   *m_xyz, *m_rad;
    long            i, j, k, l, wr, w2;
    double          prb = 1.40, vol = 0.0, ri, rj, rk;
    double          bmax[3] = { -9999., -9999., -9999. };
    double          bmin[3] = { +9999., +9999., +9999. };
    long            ci, cj, ck, wi, wj, wk;
    long            npt[3], cnt[3], *dim;
    long            cub[24] = { 1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, -1, 1, 1, -1, 1,-1, -1,-1, 1, -1,-1,-1 };
    char            ***grd, is_ok = 1;
    double          *crd, *rad, dsp, *ptr_f;
    time_t          t0;

    o_pdb = Py_False;
    o_trn = Py_False;
    if( PyArg_ParseTuple( args, "OOOd|dOO", &o_rad, &o_xyz, &o_cen, &dsp, &prb, &o_pdb, &o_trn ) ) {

        t0  = time( NULL );
        m_rad = (PyArrayObject*) PyArray_FROM_OT( o_rad, NPY_DOUBLE );
        m_xyz = (PyArrayObject*) PyArray_FROM_OT( o_xyz, NPY_DOUBLE );
        dim   = PyArray_SHAPE( m_xyz );
        rad = (double*) malloc(     dim[0] * sizeof( double ) );
        crd = (double*) malloc( 3 * dim[0] * sizeof( double ) );
fprintf(stderr,"SIZ: %ld\n",dim[0]);
fprintf(stderr,"DSP: %lf\n",dsp);
fprintf(stderr,"PRB: %lf\n",prb);
        for( i = 0; i < dim[0]; i++ ) {
            ptr_f = (double*) PyArray_GETPTR1( m_rad, i );
            rad[i] = *ptr_f;
            for( j = 0; j < 3; j++ ) {
                ptr_f = (double*) PyArray_GETPTR2( m_xyz, i, j );
                crd[3*i+j] = *ptr_f;
                bmin[j] = min( bmin[j], *ptr_f );
                bmax[j] = max( bmax[j], *ptr_f );
            }
        }
        Py_DECREF( m_rad );
        Py_DECREF( m_xyz );

        m_xyz = (PyArrayObject*) PyArray_FROM_OT( o_cen, NPY_DOUBLE );
        for( j = 0; j < 3; j++ ) { 
            ptr_f = (double*) PyArray_GETPTR1( m_xyz, j );
            npt[j] = (long)( ( bmax[j] - bmin[j] ) / dsp );
            cnt[j] = (long)( ( *ptr_f - bmin[j] ) / dsp );
        }
        Py_DECREF( m_xyz );
fprintf(stderr,"MIN: %8.3lf%8.3lf%8.3lf\n",bmin[0],bmin[1],bmin[2]);
fprintf(stderr,"MAX: %8.3lf%8.3lf%8.3lf\n",bmax[0],bmax[1],bmax[2]);
fprintf(stderr,"NPT: %8ld%8ld%8ld\n",npt[0],npt[1],npt[2]);
fprintf(stderr,"CEN: %8ld%8ld%8ld\n",cnt[0],cnt[1],cnt[2]);

        // allocate memory
        grd = malloc( sizeof( char **) * npt[0] );
        for( i = 0; i < npt[0]; i++ ) {
            grd[i] = malloc( sizeof( char *) * npt[1] );
            for( j = 0; j < npt[1]; j++ ) {
                grd[i][j] = malloc( sizeof( char ) * npt[2] );
                for( k = 0; k < npt[2]; k++ ) {
                    grd[i][j][k] = 0;
                }
            }
        }

        // init
        grd[cnt[0]][cnt[1]][cnt[2]] = 1;

        // search
        wr = (long)( ( prb + dsp ) / dsp );
        w2 = wr * wr;
        for( l = 0; l < 24; l += 3 ) {
            for( wi = cnt[0]; wi > 0 && wi < npt[0]; wi += cub[l] ) {
                ri = bmin[0] + wi * dsp;
                for( wj = cnt[1]; wj > 0 && wj < npt[1]; wj += cub[l+1] ) {
                    rj = bmin[1] + wj * dsp;
                    for( wk = cnt[2]; wk > 0 && wk < npt[2]; wk += cub[l+2] ) {
                        if( grd[wi][wj][wk] == 1 ) {
                            rk = bmin[2] + wk * dsp;
                            if( __collide( ri, rj, rk, prb, dsp, dim[0], rad, crd ) == 0 ) {
                                for( i = - wr - 1; i < wr + 1; i++ ) {
                                    ci = wi + i;
                                    if( ci >= 0 && ci < npt[0] )
                                    for( j = - wr - 1; j < wr + 1; j++ ) {
                                        cj = wj + j;
                                        if( cj >= 0 && cj < npt[1] )
                                        for( k = - wr - 1; k < wr + 1; k++ ) {
                                            ck = wk + k;
                                            if( ck >= 0 && ck < npt[2] ) 
                                                if( i * i + j * j + k * k <= w2 ) grd[ci][cj][ck] = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // prune wihtin a max volume
        if( o_trn == Py_True ) {
            char    buf[32];
            long    ss;
            double  sx, sy, sz, sr;
            FILE*   fd = fopen( "within", "rb" );

            if( fd != NULL ) {
                fread( buf, 1, 4, fd );
                memcpy( &ss, &buf[0], 4 );
fprintf(stderr,"TRN: %ld\n",ss);
                for( l = 0; l < ss; l++ ) {
                    fread( buf, 1, 32, fd );
                    memcpy( &sx,  &buf[0], 8 );
                    memcpy( &sy,  &buf[8], 8 );
                    memcpy( &sz, &buf[16], 8 );
                    memcpy( &sr, &buf[24], 8 );
                    wi = (long)( ( sx - bmin[0] ) / dsp );
                    wj = (long)( ( sy - bmin[1] ) / dsp );
                    wk = (long)( ( sz - bmin[2] ) / dsp );
                    wr = (long)( sr / dsp );
                    w2 = wr * wr;
                    for( i = - wr - 1; i < wr + 1; i++ ) {
                        ci = wi + i;
                        if( ci >= 0 && ci < npt[0] )
                            for( j = - wr - 1; j < wr + 1; j++ ) {
                                cj = wj + j;
                                if( cj >= 0 && cj < npt[1] )
                                    for( k = - wr - 1; k < wr + 1; k++ ) {
                                        ck = wk + k;
                                        if( ck >= 0 && ck < npt[2] ) 
                                            if( i * i + j * j + k * k <= w2 && grd[ci][cj][ck] == 1 ) grd[ci][cj][ck] = 2;
                                    }
                            }
                    }
                }
                fclose( fd );
                is_ok = 2;
            }
        }

        if( o_pdb == Py_True ) {
            FILE* fd = fopen( "volume.pdb", "wt" );
            for( i = 1; i < npt[0] - 1; i++ )
                for( j = 1; j < npt[1] - 1; j++ )
                    for( k = 1; k < npt[2] - 1; k++ )
                        if( grd[i][j][k] == is_ok ) { 
                            vol += 1.0;
                            if( ! ( grd[i-1][j][k] == is_ok && 
                                    grd[i+1][j][k] == is_ok && 
                                    grd[i][j-1][k] == is_ok && 
                                    grd[i][j+1][k] == is_ok && 
                                    grd[i][j][k-1] == is_ok && 
                                    grd[i][j][k+1] == is_ok ) )
                                fprintf( fd, "ATOM      1  H   SRF     1    %8.3lf%8.3lf%8.3lf  1.00  0.00    SRF\n",
                                    bmin[0] + i * dsp, bmin[1] + j * dsp, bmin[2] + k * dsp );
                        }
            fclose( fd );
        } else {
            for( i = 0; i < npt[0]; i++ )
                for( j = 0; j < npt[1]; j++ )
                    for( k = 0; k < npt[2]; k++ )
                        if( grd[i][j][k] == is_ok )
                            vol += 1.0;
        }
        vol *= dsp * dsp * dsp;
fprintf(stderr,"VOL: %lf _A^3\n",vol );

        free( rad ); free( crd );
        for( i = 0; i < npt[0]; i++ ) {
            for( j = 0; j < npt[1]; j++ ) free( grd[i][j] );
            free( grd[i] );
        }
        free( grd );

fprintf( stderr, "TIM: %ld _sec\n", time( NULL ) - t0 );
        return( Py_BuildValue( "d", vol ) );
    }
    Py_INCREF( Py_None ); return( Py_None );
}

// ------------------------------------------------------------------------------


static struct PyMethodDef methods [] = {
    { "molecular", (PyCFunction)__molecular_grid, METH_VARARGS },
    { "surface",   (PyCFunction)__molecular_surf, METH_VARARGS },
    { "cavity",    (PyCFunction)__cavity_grid,    METH_VARARGS },
    { 0, 0, 0 }
};


static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_volume",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__volume( void ) {
    PyObject    *my_module;
    my_module = PyModule_Create( &moddef );
    import_array();
    return( my_module );
}
