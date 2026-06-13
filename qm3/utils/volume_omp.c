#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

typedef struct ptn_node { long i, j, k; struct ptn_node *n; } ptn_lst;
typedef struct sph_node { long r; ptn_lst *p; struct sph_node *n; } sph_lst;
typedef struct srf_node { double x, y, z; struct srf_node *n; } srf_lst;

// ------------------------------------------------------------------------------

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

        for( i = 0; i < dim[0]; i++ ) {
            i3  = i * 3;
            ri2 = rad[i] * rad[i];
            sph = fibonacci_sphere( max( npt, lround( rad[i] * 10.0 ) ), &siz );
            flg = (long*) calloc( siz, sizeof( long ) );
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
        free( crd ); free( rad ); 

        return( (PyObject*) m_xyz );
    }
    Py_INCREF( Py_None ); return( Py_None );
}

static PyObject* __molecular_grid( PyObject *self, PyObject *args ){
    PyObject        *o_xyz, *o_rad, *o_pdb;
    PyArrayObject   *m_xyz, *m_rad;
    long            i, j, k, l, l3, cpu, *siz, rad_int, rd2, knd = 0;
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

    dsp = 0.05;
    o_pdb = Py_False;
    if( PyArg_ParseTuple( args, "lOO|dO", &cpu, &o_rad, &o_xyz, &dsp, &o_pdb ) ) {
        d3    = dsp * dsp * dsp;
        m_rad = (PyArrayObject*) PyArray_FROM_OT( o_rad, NPY_DOUBLE );
        m_xyz = (PyArrayObject*) PyArray_FROM_OT( o_xyz, NPY_DOUBLE );
        siz   = PyArray_SHAPE( m_rad );
        crd   = (double*) malloc( 3 * siz[0] * sizeof( double ) );
        
        if( cpu > 0 ) omp_set_num_threads(cpu);

        hsh = (sph_lst**) malloc( siz[0] * sizeof( sph_lst* ) );
        sph = (sph_lst*)  malloc( sizeof( sph_lst ) );
        sph->r = -1;
        sph->p = NULL;
        sph->n = NULL;
        mad = .0;
        for( l = 0; l < siz[0]; l++ ) {
            ptr_f = (double*) PyArray_GETPTR1( m_rad, l );
            mad = max( mad, *ptr_f );
            rad_int = (long)( *ptr_f / dsp );

            sph_cur = sph;
            while( sph_cur != NULL && sph_cur->r != rad_int ) { sph_cur = sph_cur->n; }
            if( sph_cur == NULL ) {
                for( sph_cur = sph; sph_cur->n != NULL; sph_cur = sph_cur->n );

                sph_cur->n = (sph_lst*) malloc( sizeof( sph_lst ) );
                sph_cur->n->r = rad_int;
                sph_cur->n->n = NULL;
                sph_cur->n->p = (ptn_lst*) malloc( sizeof( ptn_lst ) );
                sph_cur->n->p->i = 9999;
                sph_cur->n->p->j = 9999;
                sph_cur->n->p->k = 9999;
                sph_cur->n->p->n = NULL;

                ptn_cur = sph_cur->n->p;
                rd2 = rad_int * rad_int;
                for( i = -rad_int; i <= rad_int; i++ ) {
                    for( j = -rad_int; j <= rad_int; j++ ) {
                        for( k = -rad_int; k <= rad_int; k++ ) {
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

        long n01 = npt[0] * npt[1];
        long total_pts = npt[0] * npt[1] * npt[2];
        grd = (char*) calloc( total_pts, sizeof( char ) );

        #pragma omp parallel for
        for( l = 0; l < siz[0]; l++ ) {
            long l3_local = l * 3;
            long ai = (long)( ( crd[l3_local]   - bmin[0] ) / dsp );
            long aj = (long)( ( crd[l3_local+1] - bmin[1] ) / dsp );
            long ak = (long)( ( crd[l3_local+2] - bmin[2] ) / dsp );
            
            ptn_lst *c = hsh[l]->p;
            while( c != NULL ) {
                long idx = (ai + c->i) + (aj + c->j) * npt[0] + (ak + c->k) * n01;
                if( idx >= 0 && idx < total_pts ) {
                    #pragma omp atomic
                    grd[idx] += 1;
                }
                c = c->n;
            }
        }

        if( o_pdb == Py_True ) {
            FILE* fd = fopen( "volume.pdb", "wt" );
            for( i = 1; i < npt[0] - 1; i++ )
                for( j = 1; j < npt[1] - 1; j++ )
                    for( k = 1; k < npt[2] - 1; k++ ) {
                        long idx = i + j * npt[0] + k * n01;
                        if( grd[idx] > 0 ) { 
                            vol += 1.0;
                            if( ! ( grd[(i-1) + j * npt[0] + k * n01] > 0 && 
                                    grd[(i+1) + j * npt[0] + k * n01] > 0 && 
                                    grd[i + (j-1) * npt[0] + k * n01] > 0 && 
                                    grd[i + (j+1) * npt[0] + k * n01] > 0 && 
                                    grd[i + j * npt[0] + (k-1) * n01] > 0 && 
                                    grd[i + j * npt[0] + (k+1) * n01] > 0 ) )
                                fprintf( fd, "ATOM      1  H   SRF     1    %8.3lf%8.3lf%8.3lf  1.00  0.00    SRF\n",
                                    bmin[0] + i * dsp, bmin[1] + j * dsp, bmin[2] + k * dsp );
                        }
                    }
            fclose( fd );
        } else {
            #pragma omp parallel for reduction(+:vol)
            for( l = 0; l < total_pts; l++ ) {
                if( grd[l] > 0 ) vol += 1.0;
            }
        }
        vol *= d3;

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

        return( Py_BuildValue( "d", vol ) );
    }
    Py_INCREF( Py_None ); return( Py_None );
}

typedef struct { long di, dj, start_dk, end_dk; } Span;

static PyObject* __cavity_grid( PyObject *self, PyObject *args ){
    PyObject        *o_xyz, *o_rad, *o_cen, *o_pdb, *o_trn;
    PyArrayObject   *m_xyz, *m_rad;
    long            i, j, k, l, wr, w2;
    double          prb = 1.40, vol = 0.0;
    double          bmax[3] = { -9999., -9999., -9999. };
    double          bmin[3] = { +9999., +9999., +9999. };
    long            ci, cj, ck, wi, wj, wk;
    long            npt[3], cnt[3], *dim;
    long            cub[24] = { 1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, -1, 1, 1, -1, 1,-1, -1,-1, 1, -1,-1,-1 };
    char            *grd, *col_grd, is_ok = 1;
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
fprintf(stderr,"OMP: %d\n",omp_get_max_threads());
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

        long n012 = npt[0] * npt[1] * npt[2];
        long n_plane = npt[1] * npt[2];
        grd     = (char*) calloc( n012, sizeof(char) );
        col_grd = (char*) calloc( n012, sizeof(char) ); 

        // 1. RASTERIZACIÓN SÚPER RÁPIDA DE ÁTOMOS (100% MEMSET MATEMÁTICO)
        #pragma omp parallel for schedule(dynamic)
        for( long a = 0; a < dim[0]; a++ ) {
            double r = rad[a] + prb - dsp;
            if (r <= 0.0) continue;
            double r2 = r * r;
            
            long min_i = max(0L, (long)((crd[3*a] - r - bmin[0]) / dsp));
            long max_i = min(npt[0]-1, (long)((crd[3*a] + r - bmin[0]) / dsp));
            long min_j = max(0L, (long)((crd[3*a+1] - r - bmin[1]) / dsp));
            long max_j = min(npt[1]-1, (long)((crd[3*a+1] + r - bmin[1]) / dsp));
            
            for( long c_i = min_i; c_i <= max_i; c_i++ ) {
                double dx = bmin[0] + c_i * dsp - crd[3*a];
                double dx2 = dx * dx;
                for( long c_j = min_j; c_j <= max_j; c_j++ ) {
                    double dy = bmin[1] + c_j * dsp - crd[3*a+1];
                    double limit2 = r2 - (dx2 + dy * dy);
                    
                    if (limit2 >= 0.0) {
                        double limit = sqrt(limit2);
                        long start_k = max(0L, (long)ceil((crd[3*a+2] - limit - bmin[2]) / dsp));
                        long end_k   = min(npt[2]-1, (long)floor((crd[3*a+2] + limit - bmin[2]) / dsp));
                        if (end_k >= start_k) {
                            memset(&col_grd[c_i * n_plane + c_j * npt[2] + start_k], 1, end_k - start_k + 1);
                        }
                    }
                }
            }
        }

        // Init centroid
        grd[ cnt[0]*n_plane + cnt[1]*npt[2] + cnt[2] ] = is_ok;

        wr = (long)( ( prb + dsp ) / dsp );
        w2 = wr * wr;

        // 2. PRECOMPUTACIÓN DE LA ESFERA DEL FLOOD FILL (SPANS)
        Span *spans = (Span*) malloc( sizeof(Span) * (2*wr+2) * (2*wr+2) );
        long num_spans = 0;
        for( i = -wr-1; i <= wr+1; i++ ) {
            for( j = -wr-1; j <= wr+1; j++ ) {
                long limit2 = w2 - i*i - j*j;
                if (limit2 >= 0) {
                    spans[num_spans].di = i;
                    spans[num_spans].dj = j;
                    long limit = (long)sqrt(limit2);
                    spans[num_spans].start_dk = -limit;
                    spans[num_spans].end_dk = limit;
                    num_spans++;
                }
            }
        }
        
        // 3. BÚSQUEDA FLOOD-FILL OPTIMIZADA E INTELIGENTE
        for( l = 0; l < 24; l += 3 ) {
            for( wi = cnt[0]; wi > 0 && wi < npt[0]; wi += cub[l] ) {
                for( wj = cnt[1]; wj > 0 && wj < npt[1]; wj += cub[l+1] ) {
                    for( wk = cnt[2]; wk > 0 && wk < npt[2]; wk += cub[l+2] ) {
                        long idx = wi*n_plane + wj*npt[2] + wk;
                        
                        if( grd[idx] == is_ok ) {
                            // Si NO choca con ningún átomo (es 0)
                            if( col_grd[idx] == 0 ) {
                                // Truco maestro: marcarlo como visitado (2) 
                                // ¡Esto previene recalcular esta misma esfera decenas de veces!
                                col_grd[idx] = 2; 

                                // Dibujar esfera a través de Spans y Memset
                                for (long s = 0; s < num_spans; s++) {
                                    ci = wi + spans[s].di;
                                    if (ci >= 0 && ci < npt[0]) {
                                        cj = wj + spans[s].dj;
                                        if (cj >= 0 && cj < npt[1]) {
                                            long start_k = max(0L, wk + spans[s].start_dk);
                                            long end_k   = min(npt[2]-1, wk + spans[s].end_dk);
                                            if (start_k <= end_k) {
                                                memset(&grd[ci * n_plane + cj * npt[2] + start_k], is_ok, end_k - start_k + 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        free(spans);

        // prune within a max volume
        if( o_trn == Py_True ) {
            char    buf[32];
            long    ss;
            double  sx, sy, sz, sr;
            FILE* fd = fopen( "within", "rb" );

            if( fd != NULL ) {
                fread( buf, 1, 4, fd );
                memcpy( &ss, &buf[0], 4 );
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
                                        if( ck >= 0 && ck < npt[2] ) {
                                            long cidx = ci*n_plane + cj*npt[2] + ck;
                                            if( i * i + j * j + k * k <= w2 && grd[cidx] == 1 ) grd[cidx] = 2;
                                        }
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
                    for( k = 1; k < npt[2] - 1; k++ ) {
                        long idx = i*n_plane + j*npt[2] + k;
                        if( grd[idx] == is_ok ) { 
                            vol += 1.0;
                            if( ! ( grd[(i-1)*n_plane + j*npt[2] + k] == is_ok && 
                                    grd[(i+1)*n_plane + j*npt[2] + k] == is_ok && 
                                    grd[i*n_plane + (j-1)*npt[2] + k] == is_ok && 
                                    grd[i*n_plane + (j+1)*npt[2] + k] == is_ok && 
                                    grd[i*n_plane + j*npt[2] + (k-1)] == is_ok && 
                                    grd[i*n_plane + j*npt[2] + (k+1)] == is_ok ) )
                                fprintf( fd, "ATOM      1  H   SRF     1    %8.3lf%8.3lf%8.3lf  1.00  0.00    SRF\n",
                                    bmin[0] + i * dsp, bmin[1] + j * dsp, bmin[2] + k * dsp );
                        }
                    }
            fclose( fd );
        } else {
            #pragma omp parallel for reduction(+:vol)
            for( long flat_i = 0; flat_i < n012; flat_i++ ) {
                if( grd[flat_i] == is_ok ) vol += 1.0;
            }
        }
        vol *= dsp * dsp * dsp;
fprintf(stderr,"VOL: %lf _A^3\n",vol );

        
        free( rad ); free( crd );
        free( grd ); free( col_grd );

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
