#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


typedef struct {
    PyObject_HEAD
    long    natm, ni;
    long    *qm, *mm;
    double  *sc;
    double  *epsi, *rmin;
} oMMINT;


static int MMINT__init( oMMINT *self, PyObject *args, PyObject *kwds ) {
    PyObject        *o_qm = NULL, *o_mm = NULL, *o_ex = NULL, *tmp = NULL;
    PyObject        *o_mol = NULL, *o_epsi = NULL, *o_rmin = NULL;
    PyArrayObject   *m_epsi, *m_rmin;
    long            i, j, k, c, n_qm, n_mm, n_ex, *e_qm = NULL, *e_mm = NULL, i_qm, j_mm;
    double          *e_sc = NULL, *itm;

    if( PyArg_ParseTuple( args, "OOOO", &o_mol, &o_qm, &o_mm, &o_ex ) ) {

        if( PyList_Check( o_qm ) && PyList_Check( o_mm ) && PyList_Check( o_ex ) ) {

            o_epsi = PyObject_GetAttrString( o_mol, "epsi" );
            m_epsi = (PyArrayObject*) PyArray_FROM_OT( o_epsi, NPY_DOUBLE );
            o_rmin = PyObject_GetAttrString( o_mol, "rmin" );
            m_rmin = (PyArrayObject*) PyArray_FROM_OT( o_rmin, NPY_DOUBLE );
            self->natm = PyLong_AsLong( PyObject_GetAttrString( o_mol, "natm" ) );
            self->epsi = (double*) malloc( self->natm * sizeof( double ) );
            self->rmin = (double*) malloc( self->natm * sizeof( double ) );
            for( i = 0; i < self->natm; i++ ) {
                itm = (double*) PyArray_GETPTR1( m_epsi, i );
                self->epsi[i] = *itm;
                itm = (double*) PyArray_GETPTR1( m_rmin, i );
                self->rmin[i] = *itm;
            }
            Py_DECREF( m_epsi );
            Py_DECREF( m_rmin );
            Py_DECREF( o_epsi );
            Py_DECREF( o_rmin );

            n_qm = PyList_Size( o_qm );
            n_mm = PyList_Size( o_mm );
            n_ex = PyList_Size( o_ex );

            if( n_ex > 0 ) {
                e_qm = (long*) malloc( n_ex * sizeof( long ) );
                e_mm = (long*) malloc( n_ex * sizeof( long ) );
                e_sc = (double*) malloc( n_ex * sizeof( long ) );
                for( i = 0; i < n_ex; i++ ) {
                    tmp = PyList_GetItem( o_ex, i );
                    e_qm[i] = PyLong_AsLong( PyList_GetItem( tmp, 0 ) );
                    e_mm[i] = PyLong_AsLong( PyList_GetItem( tmp, 1 ) );
                    e_sc[i] = PyFloat_AsDouble( PyList_GetItem( tmp, 2 ) );
                }
            }

            self->ni = n_qm * n_mm;
            self->qm = (long*) malloc( self->ni * sizeof( long ) );
            self->mm = (long*) malloc( self->ni * sizeof( long ) );
            self->sc = (double*) malloc( self->ni * sizeof( double ) );
            c = 0;
            for( i = 0; i < n_qm; i++ ) {
                i_qm = PyLong_AsLong( PyList_GetItem( o_qm, i ) );
                for( j = 0; j < n_mm; j++ ) {
                    j_mm = PyLong_AsLong( PyList_GetItem( o_mm, j ) );
                    self->qm[c] = i_qm;
                    self->mm[c] = j_mm;
                    self->sc[c] = 1.0;
                    for( k = 0; k < n_ex; k++ )
                        if( i_qm == e_qm[k] && j_mm == e_mm[k] )
                            self->sc[c] = e_sc[k];
                    c++;
                }
            }

            free( e_qm ); free( e_mm ); free( e_sc );
        }
    }
    return( 0 );
}


static PyObject* MMINT__new( PyTypeObject *type, PyObject *args, PyObject *kwds ) {
    oMMINT    *self;

    self = (oMMINT*) type->tp_alloc( type, 0 );
    self->ni   = 0;
    self->qm   = NULL;
    self->mm   = NULL;
    self->sc   = NULL;
    self->natm = 0;
    self->epsi = NULL;
    self->rmin = NULL;
    return( (PyObject*) self ) ;
}


static void MMINT__dealloc( oMMINT *self ) {
    free( self->qm ); free( self->mm ); free( self->sc ); free( self->epsi ); free( self->rmin );
    self->ni   = 0;
    self->qm   = NULL;
    self->mm   = NULL;
    self->sc   = NULL;
    self->natm = 0;
    self->epsi = NULL;
    self->rmin = NULL;
    Py_TYPE( self )->tp_free( (PyObject*) self );
}


static PyObject* MMINT__get_grad( PyObject *self, PyObject *args ) {
    PyObject        *o_mol, *o_coor, *o_boxl, *o_grad, *o_func, *o_chrg;
    PyArrayObject   *m_coor, *m_grad, *m_boxl, *m_chrg;
    long            i, j, k;
    double          *coor = NULL, *grad = NULL, *chrg = NULL, *itm;
    double          boxl[3], dr[3], rr, r2, ss, func, df, qij, eij;
    double          EC = 1389.35484620709144110151;
    oMMINT          *obj = NULL;

    obj = (oMMINT*) self;
    if( PyArg_ParseTuple( args, "O", &o_mol ) ) {
        o_boxl = PyObject_GetAttrString( o_mol, "boxl" );
        m_boxl = (PyArrayObject*) PyArray_FROM_OT( o_boxl, NPY_DOUBLE );
        for( k = 0; k < 3; k++ ) {
            itm = (double*) PyArray_GETPTR1( m_boxl, k );
            boxl[k] = *itm;
        }
        Py_DECREF( m_boxl );
        Py_DECREF( o_boxl );

        o_coor = PyObject_GetAttrString( o_mol, "coor" );
        o_chrg = PyObject_GetAttrString( o_mol, "chrg" );
        m_coor = (PyArrayObject*) PyArray_FROM_OT( o_coor, NPY_DOUBLE );
        m_chrg = (PyArrayObject*) PyArray_FROM_OT( o_chrg, NPY_DOUBLE );
        coor = (double*) malloc( 3 * obj->natm * sizeof( double ) );
        grad = (double*) malloc( 3 * obj->natm * sizeof( double ) );
        chrg = (double*) malloc( obj->natm * sizeof( double ) );
        for( k = 0, i = 0; i < obj->natm; i++ ) {
            itm = (double*) PyArray_GETPTR1( m_chrg, i );
            chrg[i] = *itm;
            for( j = 0; j < 3; j++ ) {
                itm = (double*) PyArray_GETPTR2( m_coor, i, j );
                coor[k] = *itm;
                grad[k++] = 0.0;
            }
        }
        Py_DECREF( m_coor );
        Py_DECREF( m_chrg );
        Py_DECREF( o_coor );
        Py_DECREF( o_chrg );

        func = 0.0;
        for( i = 0; i < obj->ni; i++ ) {
            r2 = 0.0;
            for( k = 0; k < 3; k++ ) {
                dr[k] = coor[3*obj->qm[i]+k] - coor[3*obj->mm[i]+k];
//                if( dr[k] >    boxl[k] * 0.5 ) { dr[k] -= boxl[k]; }
//                if( dr[k] <= - boxl[k] * 0.5 ) { dr[k] += boxl[k]; }
                dr[k] -= boxl[k] * round( dr[k] / boxl[k] );
                r2 += dr[k] * dr[k];
            }
            rr = sqrt( r2 );
            if( rr > 0.0 ) {
                ss = ( obj->rmin[obj->qm[i]] + obj->rmin[obj->mm[i]] ) / rr;
                ss = ss * ss * ss * ss * ss * ss;
                eij = obj->epsi[obj->qm[i]] * obj->epsi[obj->mm[i]] * ss * obj->sc[i];
                qij = EC * chrg[obj->qm[i]] * chrg[obj->mm[i]] / rr * obj->sc[i];
                func += qij + eij * ( ss - 2.0 );
                df = ( 12.0 * eij * ( 1.0 - ss ) - qij ) / r2;
                for( k = 0; k < 3; k++ ) {
                    grad[3*obj->qm[i]+k] += df * dr[k];
                    grad[3*obj->mm[i]+k] -= df * dr[k];
                }
            }
        }

        o_func = PyObject_GetAttrString( o_mol, "func" );
        func  += PyFloat_AsDouble( o_func );
        Py_DECREF( o_func );
        PyObject_SetAttrString( o_mol, "func", PyFloat_FromDouble( func ) );

        o_grad = PyObject_GetAttrString( o_mol, "grad" );
        m_grad = (PyArrayObject*) PyArray_FROM_OT( o_grad, NPY_DOUBLE );
        for( k = 0, i = 0; i < obj->natm; i++ ) {
            for( j = 0; j < 3; j++ ) {
                itm = (double*) PyArray_GETPTR2( m_grad, i, j );
                *itm += grad[k++];
            }
        }
        Py_DECREF( m_grad );
        Py_DECREF( o_grad );

        free( coor );
        free( grad );
        free( chrg );
    }
    Py_INCREF( Py_None );
    return( Py_None );
}


static struct PyMethodDef MMINT_methods [] = {
    { "get_grad", (PyCFunction)MMINT__get_grad, METH_VARARGS },
    { 0, 0, 0 }
};


static struct PyMemberDef MMINT_members [] = {
    { 0, 0, 0, 0 }
};


static struct PyMethodDef methods [] = {
    { 0, 0, 0 }
};


static PyTypeObject TMMINT = {
    PyVarObject_HEAD_INIT( NULL, 0 )
    .tp_name = "MMINT",
    .tp_doc = "Truncated Non-Bonded (Electrostatics + Lennard-Jones)",
    .tp_basicsize = sizeof( oMMINT ),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = MMINT__new,
    .tp_init = (initproc) MMINT__init,
    .tp_dealloc = (destructor) MMINT__dealloc,
    .tp_members = MMINT_members,
    .tp_methods = MMINT_methods,
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_mmint",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__mmint( void ) {
    PyObject    *my_module;

    my_module = PyModule_Create( &moddef );
    PyType_Ready( &TMMINT );
    Py_INCREF( &TMMINT );
    PyModule_AddObject( my_module, "run", (PyObject *) &TMMINT );
    import_array();
    return( my_module );
}
