#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <pthread.h>


#define max(a,b) (((a)>(b))?(a):(b))


double __dist( long i3, long j3, double *xyz ) {
    return( sqrt(
            ( xyz[i3]   - xyz[j3]   ) * ( xyz[i3]   - xyz[j3]   ) +
            ( xyz[i3+1] - xyz[j3+1] ) * ( xyz[i3+1] - xyz[j3+1] ) +
            ( xyz[i3+2] - xyz[j3+2] ) * ( xyz[i3+2] - xyz[j3+2] ) ) );
}


/*
static PyObject* _coul_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *onum;
    PyArrayObject   *mcrd, *mnum, *out;
    long            *siz, dim[1], i, j, k, ww;
    double          *num, *xyz, *itm, mmm;

    if( PyArg_ParseTuple( args, "OO", &onum, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        mnum = (PyArrayObject*) PyArray_FROM_OT( onum, NPY_DOUBLE );
        num = (double*) malloc( siz[0] * sizeof( double ) );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
                itm = (double*) PyArray_GETPTR1( mnum, i );
                num[k++] = *itm;
        }
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim[0] = siz[0] * ( siz[0] + 1 ) / 2;
        out = (PyArrayObject*) PyArray_ZEROS( 1, dim, NPY_DOUBLE, 0 );
        mmm = 0.0;
        for( i = 0; i < siz[0]; i++ ) {
            k    = i * 3;
            ww   = i * siz[0] - ( ( i - 1 ) * i ) / 2;
            itm  = (double*) PyArray_GETPTR1( out, ww );
            *itm = 0.5 * pow( num[i], 2.4 );
            mmm  = max( mmm, *itm );
            for( j = i + 1; j < siz[0]; j++ ) {
                itm  = (double*) PyArray_GETPTR1( out, ww+j-i );
                *itm = num[i] * num[j] / __dist( k, j * 3, xyz );
                mmm  = max( mmm, *itm );
            }
        }
        if( mmm > 0.0 ) {
            for( i = 0; i < dim[0]; i++ ) {
                itm  = (double*) PyArray_GETPTR1( out, i );
                *itm /= mmm;
            }
        }
        free( num ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static PyObject* _coul_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *onum;
    PyArrayObject   *mcrd, *mnum, *out;
    long            *siz, dim[2], i, j, k, i3, j3, row;
    double          *num, *xyz, dr[3], r2, zz, *itm;

    if( PyArg_ParseTuple( args, "OO", &onum, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        mnum = (PyArrayObject*) PyArray_FROM_OT( onum, NPY_DOUBLE );
        num = (double*) malloc( siz[0] * sizeof( double ) );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
                itm = (double*) PyArray_GETPTR1( mnum, i );
                num[k++] = *itm;
        }
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim[0] = siz[0] * ( siz[0] + 1 ) / 2;
        dim[1] = siz[0] * 3;
        out = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
        row = 0;
        for( i = 0; i < siz[0]; i++ ) {
            i3 = i * 3;
            for( j = i; j < siz[0]; j++ ) {
                if( j != i ) {
                    j3 = j * 3;
                    dr[0] = xyz[i3]   - xyz[j3];
                    dr[1] = xyz[i3+1] - xyz[j3+1];
                    dr[2] = xyz[i3+2] - xyz[j3+2];
                    r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    zz = num[i] * num[j] / ( r2 * sqrt( r2 ) );
                    for( k = 0; k < 3; k ++ ) {
                        itm  = (double*) PyArray_GETPTR2( out, row, i3+k );
                        *itm = - dr[k] * zz;
                        itm  = (double*) PyArray_GETPTR2( out, row, j3+k );
                        *itm =   dr[k] * zz;
                    }
                }
                row++;
            }
        }
        free( num ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}
*/


static PyObject* _coul_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *onum;
    PyArrayObject   *mcrd, *mnum, *out;
    long            *siz, dim[1], i, j, k, ww;
    double          *num, *xyz, *itm;

    if( PyArg_ParseTuple( args, "OO", &onum, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        mnum = (PyArrayObject*) PyArray_FROM_OT( onum, NPY_DOUBLE );
        num = (double*) malloc( siz[0] * sizeof( double ) );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
                itm = (double*) PyArray_GETPTR1( mnum, i );
                num[k++] = *itm;
        }
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim[0] = siz[0] * ( siz[0] - 1 ) / 2;
        out = (PyArrayObject*) PyArray_ZEROS( 1, dim, NPY_DOUBLE, 0 );
        for( i = 0; i < siz[0] - 1; i++ ) {
            k    = i * 3;
            ww   = i * siz[0] - ( ( i + 1 ) * i ) / 2 - i - 1;
            for( j = i + 1; j < siz[0]; j++ ) {
                itm  = (double*) PyArray_GETPTR1( out, ww+j );
                *itm = num[i] * num[j] / __dist( k, j * 3, xyz );
            }
        }
        free( num ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


/*
static PyObject* _coul_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *onum;
    PyArrayObject   *mcrd, *mnum, *out;
    long            *siz, dim[2], i, j, k, i3, j3, row;
    double          *num, *xyz, dr[3], r2, zz, *itm;

    if( PyArg_ParseTuple( args, "OO", &onum, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        mnum = (PyArrayObject*) PyArray_FROM_OT( onum, NPY_DOUBLE );
        num = (double*) malloc( siz[0] * sizeof( double ) );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
                itm = (double*) PyArray_GETPTR1( mnum, i );
                num[k++] = *itm;
        }
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim[0] = siz[0] * ( siz[0] - 1 ) / 2;
        dim[1] = siz[0] * 3;
        out = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
        row = 0;
        for( i = 0; i < siz[0] - 1; i++ ) {
            i3 = i * 3;
            for( j = i + 1; j < siz[0]; j++ ) {
                j3 = j * 3;
                dr[0] = xyz[i3]   - xyz[j3];
                dr[1] = xyz[i3+1] - xyz[j3+1];
                dr[2] = xyz[i3+2] - xyz[j3+2];
                r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                zz = num[i] * num[j] / ( r2 * sqrt( r2 ) );
                for( k = 0; k < 3; k ++ ) {
                    itm  = (double*) PyArray_GETPTR2( out, row, i3+k );
                    *itm = - dr[k] * zz;
                    itm  = (double*) PyArray_GETPTR2( out, row, j3+k );
                    *itm =   dr[k] * zz;
                }
                row++;
            }
        }
        free( num ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}
*/


double __cosang( long i3, long j3, long k3, double *xyz ) {
    double    vij[3], vik[3], mij, mik;
    vij[0] = xyz[j3]   - xyz[i3];
    vij[1] = xyz[j3+1] - xyz[i3+1];
    vij[2] = xyz[j3+2] - xyz[i3+2];
    mij    = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
    vik[0] = xyz[k3]   - xyz[i3];
    vik[1] = xyz[k3+1] - xyz[i3+1];
    vik[2] = xyz[k3+2] - xyz[i3+2];
    mik    = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];
    return( ( vij[0] * vik[0] + vij[1] * vik[1] + vij[2] * vik[2] ) / sqrt( mij * mik ) );
}


double __fcut( double dst, double cut ) {
    if( dst > cut ) { return( 0.0 ); } else { return( 0.5 * ( cos( M_PI * dst / cut ) + 1.0 ) ); }
}


static PyObject* _acsf_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *oeta2, *oeta5;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim, odim[1], i, i3, j, j3, k, k3, l, dd, neta2, neta5;
    double          *xyz, cutx, dse5, pre5, *eta2, *eta5, fij, fik, dij, dik, *itm, mx2, mx5;

    if( PyArg_ParseTuple( args, "dOdOO", &cutx, &oeta2, &dse5, &oeta5, &ocrd ) ) {
        neta2 = PyList_Size( oeta2 );
        neta5 = PyList_Size( oeta5 );
        eta2  = (double*) malloc( neta2 * sizeof( double ) );
        eta5  = (double*) malloc( neta5 * sizeof( double ) );
        for( i = 0; i < neta2; i++ ) eta2[i] = PyFloat_AsDouble( PyList_GetItem( oeta2, i ) );
        for( i = 0; i < neta5; i++ ) eta5[i] = PyFloat_AsDouble( PyList_GetItem( oeta5, i ) );
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz   = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim     = neta2 + neta5;
        odim[0] = siz[0] * dim;
        out     = (PyArrayObject*) PyArray_ZEROS( 1, odim, NPY_DOUBLE, 0 );
        pre5    = pow( 2.0, 1.0 - dse5 );
        for( i = 0; i < siz[0]; i++ ) {
            i3 = i * 3;
            dd = i * dim;
            for( j = 0; j < siz[0]; j++ ) if( j != i ) {
                j3  = j * 3;
                dij = __dist( i3, j3, xyz );
                fij = __fcut( dij, cutx );
                if( fij > 0.0 ) {
                    for( l = 0; l < neta2; l++ ) { 
                        itm  = (double*) PyArray_GETPTR1( out, dd + l );
                        *itm += fij * exp( - eta2[l] * dij * dij );
                    }
                    for( k = 0; k < siz[0]; k++ ) if( k != j && k != i ) {
                        k3  = k * 3;
                        dik = __dist( i3, k3, xyz );
                        fik = __fcut( dik, cutx );
                        if( fik > 0.0 ) {
                            for( l = 0; l < neta5; l++ ) {
                                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + l );
                                *itm += pre5 * fij * fik * pow( 1.0 + __cosang( i3, j3, k3, xyz ), dse5 ) * exp( - eta5[l] * ( dij * dij + dik * dik ) );
                            }
                        }
                    }
                }
            }
        }
//        // normalize by kind ---------------------------------------------
//        mx2 = 0.0;
//        mx5 = 0.0;
//        for( i = 0; i < siz[0]; i++ ) {
//            dd = i * dim;
//            for( l = 0; l < neta2; l++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + l );
//                mx2 = max( mx2, *itm );
//            }
//            for( l = 0; l < neta5; l++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + l );
//                mx5 = max( mx5, *itm );
//            }
//        }
//        if( mx2 < 1.e-6 ) mx2 = 1.0;
//        if( mx5 < 1.e-6 ) mx5 = 1.0;
//        for( i = 0; i < siz[0]; i++ ) {
//            dd = i * dim;
//            for( l = 0; l < neta2; l++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + l );
//                *itm /= mx2;
//            }
//            for( l = 0; l < neta5; l++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + l );
//                *itm /= mx5;
//            }
//        }
//        // ---------------------------------------------------------------
        free( eta2 ); free( eta5 ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}
 

typedef struct { long ini, end, siz, dim; double *xyz, *out; long neta2, neta5; double cutx, pre5, dse5, *eta2, *eta5; } p_arg;

void* __acsf( void *args ) {
    p_arg       *arg = (p_arg*) args;
    long        i, i3, j, j3, k, k3, l, dd;
    double      fij, fik, dij, dik;

    for( i = 0; i < arg->dim * arg->siz; i++ ) arg->out[i] = 0.0;
    for( i = arg->ini; i < arg->end; i++ ) {
        i3 = i * 3;
        dd = i * arg->dim;
        for( j = 0; j < arg->siz; j++ ) if( j != i ) {
            j3  = j * 3;
            dij = __dist( i3, j3, arg->xyz );
            fij = __fcut( dij, arg->cutx );
            if( fij > 0.0 ) {
                for( l = 0; l < arg->neta2; l++ ) { 
                    arg->out[dd+l] += fij * exp( - arg->eta2[l] * dij * dij );
                }
                for( k = 0; k < arg->siz; k++ ) if( k != j && k != i ) {
                    k3  = k * 3;
                    dik = __dist( i3, k3, arg->xyz );
                    fik = __fcut( dik, arg->cutx );
                    if( fik > 0.0 ) {
                        for( l = 0; l < arg->neta5; l++ ) {
                            arg->out[dd+arg->neta2+l] += arg->pre5 * fij * fik * pow( 1.0 + __cosang( i3, j3, k3, arg->xyz ), arg->dse5 ) * exp( - arg->eta5[l] * ( dij * dij + dik * dik ) );
                        }
                    }
                }
            }
        }
    }
}

static PyObject* _acsf_pinf( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *oeta2, *oeta5;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim, odim[1], i, j, k, dd, neta2, neta5, ncpu = 2;
    double          *xyz, cutx, dse5, pre5, *eta2, *eta5, *itm, mx2, mx5;
    pthread_t       *pid;
    p_arg           *arg;

    if( PyArg_ParseTuple( args, "dOdOO|l", &cutx, &oeta2, &dse5, &oeta5, &ocrd, &ncpu ) ) {
        neta2 = PyList_Size( oeta2 );
        neta5 = PyList_Size( oeta5 );
        eta2  = (double*) malloc( neta2 * sizeof( double ) );
        eta5  = (double*) malloc( neta5 * sizeof( double ) );
        for( i = 0; i < neta2; i++ ) eta2[i] = PyFloat_AsDouble( PyList_GetItem( oeta2, i ) );
        for( i = 0; i < neta5; i++ ) eta5[i] = PyFloat_AsDouble( PyList_GetItem( oeta5, i ) );
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz   = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        dim     = neta2 + neta5;
        pre5    = pow( 2.0, 1.0 - dse5 );
        odim[0] = siz[0] * dim;
        out     = (PyArrayObject*) PyArray_ZEROS( 1, odim, NPY_DOUBLE, 0 );
        
        pid = (pthread_t*) malloc( ncpu * sizeof( pthread_t ) );
        arg = (p_arg*) malloc( ncpu * sizeof( p_arg ) );
        for( j = siz[0] / ncpu, i = 0; i < ncpu; i++ ) { 
            arg[i].ini = i * j;
            arg[i].end = ( i + 1 ) * j;
        }
        arg[ncpu-1].end += siz[0] % ncpu;
        for( i = 0; i < ncpu; i++ ) { 
            arg[i].siz   = siz[0];
            arg[i].dim   = dim;
            arg[i].xyz   = xyz;
            arg[i].out   = (double*) malloc( odim[0] * sizeof( double ) );
            arg[i].cutx  = cutx;
            arg[i].neta2 = neta2;
            arg[i].neta5 = neta5;
            arg[i].dse5  = dse5;
            arg[i].pre5  = pre5;
            arg[i].eta2  = eta2;
            arg[i].eta5  = eta5;
            pthread_create( &pid[i], NULL, __acsf, (void*) &arg[i] );
        }
        for( i = 0; i < ncpu; i++ ) pthread_join( pid[i], NULL );
        // backup
//        mx2 = 0.0;
//        mx5 = 0.0;
        for( j = 0; j < siz[0]; j++ ) {
            dd = j * dim;
            for( k = 0; k < neta2; k++ ) {
                itm  = (double*) PyArray_GETPTR1( out, dd + k );
                for( i = 0; i < ncpu; i++ ) *itm += arg[i].out[dd + k];
//                mx2 = max( mx2, *itm );
            }
            for( k = 0; k < neta5; k++ ) {
                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + k );
                for( i = 0; i < ncpu; i++ ) *itm += arg[i].out[dd + neta2 + k];
//                mx5 = max( mx5, *itm );
            }
        }
        free( arg ); free( pid );
//        // normalize
//        if( mx2 < 1.e-6 ) mx2 = 1.0;
//        if( mx5 < 1.e-6 ) mx5 = 1.0;
//        for( j = 0; j < siz[0]; j++ ) {
//            dd = j * dim;
//            for( k = 0; k < neta2; k++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + k );
//                *itm /= mx2;
//            }
//            for( k = 0; k < neta5; k++ ) {
//                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + k );
//                *itm /= mx5;
//            }
//        }
        free( eta2 ); free( eta5 ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static struct PyMethodDef methods [] = {
    { "coul_info", (PyCFunction)_coul_info, METH_VARARGS },
//    { "coul_jaco", (PyCFunction)_coul_jaco, METH_VARARGS },
    { "acsf_info", (PyCFunction)_acsf_info, METH_VARARGS },
    { "acsf_pinf", (PyCFunction)_acsf_pinf, METH_VARARGS },
    { 0, 0, 0 }
};


static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_ml_info",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__ml_info( void ) {
    PyObject    *my_module;
    my_module = PyModule_Create( &moddef );
    import_array();
    return( my_module );
}
