#include <Python.h>
#include "structmember.h"
#include <stdio.h>
#include <math.h>
#include "Plumed.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


typedef struct {
    PyObject_HEAD
    int       Natoms, box;
    double    *Masses;
    plumed    pmed;
} oPLUMED;


static int __init( oPLUMED *self, PyObject *args, PyObject *kwds ) {
    PyObject        *mol, *omas, *pbc = Py_False;
    PyArrayObject   *mmas;
    long            i;
    int             RealPrecision = 8;
    double          *ptr, MDLengthUnits = 0.1, MDChargeUnits = 1.602176565e-19, MDTimeUnits = 0.001, Timestep = 1.0;

    if( PyArg_ParseTuple( args, "O|Od", &mol, &pbc, &MDTimeUnits ) ) {

        if( pbc == Py_True ) { self->box = 1; } else { self->box = 0; }
        self->Natoms = (int) PyLong_AsLong( PyObject_GetAttrString( mol, "natm" ) );
        self->Masses = (double*) malloc( self->Natoms * sizeof( double ) );
        omas = PyObject_GetAttrString( mol, "mass" );
        mmas = (PyArrayObject*) PyArray_FROM_OT( omas, NPY_DOUBLE );
        for( i = 0; i < self->Natoms; i++ ) {
            ptr = (double*) PyArray_GETPTR1( mmas, i );
            self->Masses[i] = *ptr;
        }
        Py_DECREF( mmas );
        Py_DECREF( omas );

        self->pmed = plumed_create();
        // double
        plumed_cmd( self->pmed, "setRealPrecision", &(RealPrecision) );
        // Angstrom to nanometers
        plumed_cmd( self->pmed, "setMDLengthUnits", &(MDLengthUnits) );
        // MD time units
        plumed_cmd( self->pmed, "setMDTimeUnits", &(MDTimeUnits) );
        // Time step
        plumed_cmd( self->pmed, "setTimestep", &(Timestep) );
        // atomic to coulomb
        plumed_cmd( self->pmed, "setMDChargeUnits", &(MDChargeUnits) );
        // plumed config name
        plumed_cmd( self->pmed, "setPlumedDat", "plumed.dat" );
        // number of atmos
        plumed_cmd( self->pmed, "setNatoms", &(self->Natoms) );
        // log filename
        plumed_cmd( self->pmed, "setLogFile", "plumed.log" );
        // skip virial
        plumed_cmd( self->pmed, "setNoVirial", NULL );
        // go!
        plumed_cmd( self->pmed, "init", NULL );
    }
    return( 0 );
}


static PyObject* __new( PyTypeObject *type, PyObject *args, PyObject *kwds ) {
    oPLUMED    *self;
    int        i;

    self = (oPLUMED*) type->tp_alloc( type, 0 );
    self->Masses = NULL;
    return( (PyObject*) self ) ;
}


static void __dealloc( oPLUMED *self ) {
    plumed_finalize( self->pmed );
    free( self->Masses );
    Py_TYPE( self )->tp_free( (PyObject*) self );
}


static PyObject* __calc( PyObject *self, PyObject *args ) {
    PyObject        *mol, *otmp;
    PyArrayObject   *mtmp;
    long            i, j, stp = 0;
    double          *xyz, *frz, *ptr, ene, acc = 0.0;
    double          box[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    oPLUMED         *obj = NULL;

    obj = (oPLUMED*) self;
    if( PyArg_ParseTuple( args, "O|l", &mol, &stp ) ) {

        if( obj->box == 1 ) {
            // box latice vectors in nanometers...
            otmp = PyObject_GetAttrString( mol, "boxl" );
            mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
            for( i = 0; i < 3; i++ ) {
                ptr = (double*) PyArray_GETPTR1( mtmp, i );
                box[3*i+i] = (*ptr) * 0.1;
            }
            Py_DECREF( mtmp );
            Py_DECREF( otmp );
        }

        otmp = PyObject_GetAttrString( mol, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        xyz = (double*) malloc( 3 * obj->Natoms * sizeof( double ) );
        frz = (double*) malloc( 3 * obj->Natoms * sizeof( double ) );
        for( i = 0; i < obj->Natoms; i++ ) {
            for( j = 0; j < 3; j++ ) {
                ptr = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[3*i+j] = *ptr;
                frz[3*i+j] = 0.0;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( mol, "func" );
        ene = PyFloat_AsDouble( otmp );
        Py_DECREF( otmp );

        plumed_cmd( obj->pmed, "setStepLong", &(stp) );
        plumed_cmd( obj->pmed, "setPositions", xyz );
        plumed_cmd( obj->pmed, "setMasses", obj->Masses );

        // charge based constraints...
        //plumed_cmd( obj->pmed, "setCharges", ??? );

        if( obj->box == 1 ) plumed_cmd( obj->pmed, "setBox", &box[0] );

        // energy based constraints...
        //plumed_cmd( obj->pmed, "setEnergy", &ene );

        plumed_cmd( obj->pmed, "setForces", frz );
        plumed_cmd( obj->pmed, "calc", NULL );
        plumed_cmd( obj->pmed, "getBias", &acc );

        PyObject_SetAttrString( mol, "func", PyFloat_FromDouble( ene + acc ) );

        otmp = PyObject_GetAttrString( mol, "grad" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < obj->Natoms; i++ ) {
            for( j = 0; j < 3; j++ ) {
                ptr = (double*) PyArray_GETPTR2( mtmp, i, j );
                *ptr -= frz[3*i+j];
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        free( xyz ); free( frz );
        
    }
    Py_INCREF( Py_None );
    return( Py_None );
}


static struct PyMethodDef PLUMED_methods [] = {
    { "get_grad", (PyCFunction)__calc, METH_VARARGS },
    { 0, 0, 0 }
};

static struct PyMethodDef methods [] = {
    { 0, 0, 0 }
};

static struct PyMemberDef members [] = {
    { 0, 0, 0, 0 }
};


// --------------------------------------------------------------------------------------


static PyTypeObject tPLUMED = {
    PyVarObject_HEAD_INIT( NULL, 0 )
    .tp_name = "Plumed",
    .tp_doc = "Plumed object",
    .tp_basicsize = sizeof( oPLUMED ),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = __new,
    .tp_init = (initproc) __init,
    .tp_dealloc = (destructor) __dealloc,
    .tp_members = members,
    .tp_methods = PLUMED_methods,
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_plumed",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__plumed( void ) {
    PyObject    *my_module;

    my_module = PyModule_Create( &moddef );
    PyType_Ready( &tPLUMED );
    Py_INCREF( &tPLUMED );
    PyModule_AddObject( my_module, "run", (PyObject *) &tPLUMED );
    import_array();
    return( my_module );
}
