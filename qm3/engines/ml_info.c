#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <pthread.h>


double __dist( long i3, long j3, double *xyz ) {
    return( sqrt(
            ( xyz[j3]   - xyz[i3]   ) * ( xyz[j3]   - xyz[i3]   ) +
            ( xyz[j3+1] - xyz[i3+1] ) * ( xyz[j3+1] - xyz[i3+1] ) +
            ( xyz[j3+2] - xyz[i3+2] ) * ( xyz[j3+2] - xyz[i3+2] ) ) );
}


static PyObject* _coul_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim[1], i, j, k, ww;
    double          *xyz, *itm;

    if( PyArg_ParseTuple( args, "O", &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mcrd );
        dim[0] = siz[0] * ( siz[0] - 1 ) / 2;
        out = (PyArrayObject*) PyArray_ZEROS( 1, dim, NPY_DOUBLE, 0 );
        for( i = 0; i < siz[0] - 1; i++ ) {
            k    = i * 3;
            ww   = i * siz[0] - ( ( i + 1 ) * i ) / 2 - i - 1;
            for( j = i + 1; j < siz[0]; j++ ) {
                itm  = (double*) PyArray_GETPTR1( out, ww+j );
                *itm = 1.0 / __dist( k, j * 3, xyz );
            }
        }
        free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static PyObject* _coul_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim[2], i, j, k, i3, j3, row;
    double          *xyz, dr[3], r2, zz, *itm;

    if( PyArg_ParseTuple( args, "O", &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mcrd );
        dim[0] = siz[0] * ( siz[0] - 1 ) / 2;
        dim[1] = siz[0] * 3;
        out = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
        row = 0;
        for( i = 0; i < siz[0] - 1; i++ ) {
            i3 = i * 3;
            for( j = i + 1; j < siz[0]; j++ ) {
                j3 = j * 3;
                dr[0] = xyz[j3]   - xyz[i3];
                dr[1] = xyz[j3+1] - xyz[i3+1];
                dr[2] = xyz[j3+2] - xyz[i3+2];
                r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                zz = 1.0 / ( r2 * sqrt( r2 ) );
                for( k = 0; k < 3; k ++ ) {
                    itm  = (double*) PyArray_GETPTR2( out, row, i3+k );
                    *itm =   dr[k] * zz;
                    itm  = (double*) PyArray_GETPTR2( out, row, j3+k );
                    *itm = - dr[k] * zz;
                }
                row++;
            }
        }
        free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static PyObject* _bbnd_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *obag;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim[1], i, j, k;
    double          *xyz, *itm;

    if( PyArg_ParseTuple( args, "OO", &obag, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mcrd );

		dim[0] = (long)( PyList_Size( obag ) / 2 );
        out = (PyArrayObject*) PyArray_ZEROS( 1, dim, NPY_DOUBLE, 0 );
		for( i = 0; i < dim[0]; i++ ) {
			j = PyLong_AsLong( PyList_GetItem( obag, 2 * i     ) ) * 3;
			k = PyLong_AsLong( PyList_GetItem( obag, 2 * i + 1 ) ) * 3;
            itm  = (double*) PyArray_GETPTR1( out, i );
            *itm = 1.0 / __dist( k, j, xyz );
		}

        free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


/*
static PyObject* _bbnd_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *obag;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim[2], i, j, k, l;
    double          *xyz, dr[3], r2, zz, *itm;

    if( PyArg_ParseTuple( args, "OO", &obag, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mcrd );

		dim[0] = (long)( PyList_Size( obag ) / 2 );
        dim[1] = siz[0] * 3;
        out = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
		for( i = 0; i < dim[0]; i++ ) {
			j = PyLong_AsLong( PyList_GetItem( obag, 2 * i     ) ) * 3;
			k = PyLong_AsLong( PyList_GetItem( obag, 2 * i + 1 ) ) * 3;
            dr[0] = xyz[j]   - xyz[k];
            dr[1] = xyz[j+1] - xyz[k+1];
            dr[2] = xyz[j+2] - xyz[k+2];
            r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            zz = 1.0 / ( r2 * sqrt( r2 ) );
            for( l = 0; l < 3; l ++ ) {
                itm  = (double*) PyArray_GETPTR2( out, i, j+l );
                *itm =   dr[l] * zz;
                itm  = (double*) PyArray_GETPTR2( out, i, k+l );
                *itm = - dr[l] * zz;
            }
		}

        free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}
*/


static PyObject* _bbnd_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *obag;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim[2], i, j, k, l;
    double          *xyz, dr[3], r2, zz, *itm;

    if( PyArg_ParseTuple( args, "OO", &obag, &ocrd ) ) {
        mcrd = (PyArrayObject*) PyArray_FROM_OT( ocrd, NPY_DOUBLE );
        siz = PyArray_SHAPE( mcrd );
        xyz = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mcrd, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mcrd );

		dim[0] = (long)( PyList_Size( obag ) / 2 );
        dim[1] = 3;
        out = (PyArrayObject*) PyArray_ZEROS( 2, dim, NPY_DOUBLE, 0 );
		for( i = 0; i < dim[0]; i++ ) {
			j = PyLong_AsLong( PyList_GetItem( obag, 2 * i     ) ) * 3;
			k = PyLong_AsLong( PyList_GetItem( obag, 2 * i + 1 ) ) * 3;
            dr[0] = xyz[j]   - xyz[k];
            dr[1] = xyz[j+1] - xyz[k+1];
            dr[2] = xyz[j+2] - xyz[k+2];
            r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            zz = 1.0 / ( r2 * sqrt( r2 ) );
            for( l = 0; l < 3; l ++ ) {
                itm  = (double*) PyArray_GETPTR2( out, i, l );
                *itm =   dr[l] * zz;
            }
		}

        free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static struct PyMethodDef methods [] = {
    { "coul_info", (PyCFunction)_coul_info, METH_VARARGS },
    { "coul_jaco", (PyCFunction)_coul_jaco, METH_VARARGS },

    { "bbnd_info", (PyCFunction)_bbnd_info, METH_VARARGS },
    { "bbnd_jaco", (PyCFunction)_bbnd_jaco, METH_VARARGS },

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
