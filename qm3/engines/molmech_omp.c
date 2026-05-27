#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

typedef struct two_lst_node { long i,j; struct two_lst_node *n; } two_lst;

// ####################################################################################################################

static PyObject* w_guess_angles( PyObject *self, PyObject *args ) {
    PyObject    *out, *object, *otmp, *olst;
    long        i, j, k, *lst, siz, cnt, ii, jj, kk;
    long        nat, **con, *nel;

    if( PyArg_ParseTuple( args, "O", &object ) ) {

        otmp = PyObject_GetAttrString( object, "bond" );
        siz  = PyList_Size( otmp );
        lst  = (long*) malloc( 2*siz * sizeof( long ) );
        for( i = 0; i < siz; i++ )
            for( j = 0; j < 2; j++ ) lst[2*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        nat  = PyLong_AsLong( PyObject_GetAttrString( object, "natm" ) );
        otmp = PyObject_GetAttrString( object, "conn" );
        con  = (long**) malloc( nat * sizeof( long* ) );
        nel  = (long*) malloc( nat * sizeof( long ) );
        for( i = 0 ; i < nat; i++ ) {
            olst = PyList_GetItem( otmp, i );
            nel[i] = PyList_Size( olst );
            con[i] = (long*) malloc( nel[i] * sizeof( long ) );
            for( j = 0; j < nel[i]; j++ ) con[i][j] = PyLong_AsLong( PyList_GetItem( olst, j ) );
        }
        Py_DECREF( otmp );
        out = PyList_New( 0 );
        for( i = 0; i < siz - 1; i++ ) {
            for( j = i + 1; j < siz; j++ ) {
                ii = -1; jj = -1; kk = -1;
                if( lst[2*i] == lst[2*j] ) {
                    ii = lst[2*i+1]; jj = lst[2*i]; kk = lst[2*j+1];
                } else if( lst[2*i] == lst[2*j+1] ) {
                    ii = lst[2*i+1]; jj = lst[2*i]; kk = lst[2*j];
                } else if( lst[2*i+1] == lst[2*j] ) {
                    ii = lst[2*i]; jj = lst[2*i+1]; kk = lst[2*j+1];
                } else if( lst[2*i+1] == lst[2*j+1] ) {
                    ii = lst[2*i]; jj = lst[2*i+1]; kk = lst[2*j];
                }
                if( ii != -1 && jj != -1 && kk != -1 ) {
                    cnt = 0; for( k = 0; k < nel[ii]; k++ ) cnt += ( kk == con[ii][k] );
                    if( cnt == 0 ) { 
                        otmp = Py_BuildValue( "[l,l,l]", ii, jj, kk );
                        PyList_Append( out, otmp );
                        Py_DECREF( otmp );
                    }
                }
            }
        }
        free( lst ); free( nel ); for( i = 0; i < nat; i++ ) free( con[i] ); free( con );
        return( out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_guess_dihedrals( PyObject *self, PyObject *args ) {
    PyObject    *out, *object, *otmp, *olst;
    long        i, j, k, ii, jj, kk, ll, *lst, siz, cnt;
    long        nat, **con, *nel;

    if( PyArg_ParseTuple( args, "O", &object ) ) {

        otmp = PyObject_GetAttrString( object, "angl" );
        siz  = PyList_Size( otmp );
        lst  = (long*) malloc( 3*siz * sizeof( long ) );
        for( i = 0; i < siz; i++ )
            for( j = 0; j < 3; j++ ) lst[3*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        nat  = PyLong_AsLong( PyObject_GetAttrString( object, "natm" ) );
        otmp = PyObject_GetAttrString( object, "conn" );
        con  = (long**) malloc( nat * sizeof( long* ) );
        nel  = (long*) malloc( nat * sizeof( long ) );
        for( i = 0 ; i < nat; i++ ) {
            olst = PyList_GetItem( otmp, i );
            nel[i] = PyList_Size( olst );
            con[i] = (long*) malloc( nel[i] * sizeof( long ) );
            for( j = 0; j < nel[i]; j++ ) con[i][j] = PyLong_AsLong( PyList_GetItem( olst, j ) );
        }
        Py_DECREF( otmp );
        out = PyList_New( 0 );
        for( i = 0; i < siz - 1; i++ ) {
            for( j = i + 1; j < siz; j++ ) {
                ii = -1; jj = -1; kk = -1; ll = -1;
                if( lst[3*i+1] == lst[3*j] && lst[3*i+2] == lst[3*j+1] ) {
                    ii = lst[3*i]; jj = lst[3*i+1]; kk = lst[3*i+2]; ll = lst[3*j+2];
                } else if( lst[3*i+1] == lst[3*j+2] && lst[3*i+2] == lst[3*j+1] ) {
                    ii = lst[3*i]; jj = lst[3*i+1]; kk = lst[3*i+2]; ll = lst[3*j];
                } else if( lst[3*i+1] == lst[3*j] && lst[3*i] == lst[3*j+1] ) {
                    ii = lst[3*i+2]; jj = lst[3*i+1]; kk = lst[3*i]; ll = lst[3*j+2];
                } else if( lst[3*i+1] == lst[3*j+2] && lst[3*i] == lst[3*j+1] ) {
                    ii = lst[3*i+2]; jj = lst[3*i+1]; kk = lst[3*i]; ll = lst[3*j];
                }
                if( ii != -1 && jj != -1 && kk != -1 && ll != -1 ) {
                    cnt = 0; for( k = 0; k < nel[ii]; k++ ) cnt += ( ll == con[ii][k] );
                    if( cnt == 0 ) {
                        otmp = Py_BuildValue( "[l,l,l,l]", ii, jj, kk, ll );
                        PyList_Append( out, otmp );
                        Py_DECREF( otmp );
                    }
                }
            }
        }
        free( lst ); free( nel ); for( i = 0; i < nat; i++ ) free( con[i] ); free( con );
        return( out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_update_non_bonded( PyObject *self, PyObject *args ) {
    PyObject        *out, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, cut, box[3], *itm;
    long            *bnd, *ang, *dih, n_bnd, n_ang, n_dih;
    long            i, j, k, cpu, *siz, *qms, *fre;
    two_lst         **thread_nbn;

    if( PyArg_ParseTuple( args, "OO", &object, &molecule ) ) {
        otmp = PyObject_GetAttrString( object, "ncpu" );
        cpu  = PyLong_AsLong( otmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "cut_list" );
        cut = PyFloat_AsDouble( otmp );
        if( cut > 0.0 ) { cut *= cut; } else { cut = 1.0e99; }
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "boxl" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < 3; i++ ) {
            itm = (double*) PyArray_GETPTR1( mtmp, i );
            box[i] = *itm;
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "qmat" );
        qms  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) qms[i] = ( Py_True == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = ( Py_True == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "bond" );
        n_bnd = PyList_Size( otmp );
        bnd   = (long*) malloc( 2*n_bnd * sizeof( long ) );
        for( i = 0; i < n_bnd; i++ ) {
            bnd[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            bnd[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 1 ) );
        }
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "angl" );
        n_ang = PyList_Size( otmp );
        ang   = (long*) malloc( 2*n_ang * sizeof( long ) );
        for( i = 0; i < n_ang; i++ ) {
            ang[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            ang[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 2 ) );
        }
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "dihe" );
        n_dih = PyList_Size( otmp );
        dih   = (long*) malloc( 2*n_dih * sizeof( long ) );
        for( i = 0; i < n_dih; i++ ) {
            dih[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            dih[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 3 ) );
        }
        Py_DECREF( otmp );

        omp_set_num_threads(cpu);
        thread_nbn = (two_lst**) malloc( cpu * sizeof( two_lst* ) );
        for( i = 0; i < cpu; i++ ) {
            thread_nbn[i] = (two_lst*) malloc( sizeof( two_lst ) );
            thread_nbn[i]->n = NULL;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            two_lst *pt2 = thread_nbn[tid];
            long _i, _j, _k, i3, j3, f;
            double r2, dr;

            #pragma omp for schedule(guided)
            for( _i = 0; _i < siz[0] - 1; _i++ ) {
                for( _j = _i + 1; _j < siz[0]; _j++ ) {
                    if( ( qms[_i] == 1 && qms[_j] == 1 ) || ( fre[_i] == 0 && fre[_j] == 0 ) ) { continue; }
                    i3  = 3 * _i;
                    j3  = 3 * _j;
                    r2 = 0.0;
                    for( _k = 0; _k < 3; _k++ ) {
                        dr  = xyz[i3+_k] - xyz[j3+_k];
                        dr -= box[_k] * round( dr / box[_k] );
                        r2 += dr * dr;
                    }
                    if( r2 <= cut ) {
                        f = 0;
                        _k = 0;
                        while( _k < n_bnd && f == 0 ) {
                            f |= ( ( _i == bnd[2*_k] && _j == bnd[2*_k+1] ) || ( _i == bnd[2*_k+1] && _j == bnd[2*_k] ) );
                            _k++;
                        }
                        _k = 0;
                        while( _k < n_ang && f == 0 ) {
                            f |= ( ( _i == ang[2*_k] && _j == ang[2*_k+1] ) || ( _i == ang[2*_k+1] && _j == ang[2*_k] ) );
                            _k++;
                        }
                        _k = 0;
                        while( _k < n_dih && f == 0 ) {
                            f |= ( ( _i == dih[2*_k] && _j == dih[2*_k+1] ) || ( _i == dih[2*_k+1] && _j == dih[2*_k] ) );
                            _k++;
                        }
                        if( f == 0 ) {
                            pt2->n = (two_lst*) malloc( sizeof( two_lst ) );
                            pt2->n->i = _i;
                            pt2->n->j = _j;
                            pt2->n->n = NULL;
                            pt2 = pt2->n;
                        }
                    }
                }
            }
        }

        // Contar el tamaño total necesario
        long total_size = n_dih;
        for( i = 0; i < cpu; i++ ) {
            two_lst *pt2 = thread_nbn[i]->n;
            while( pt2 != NULL ) {
                total_size++;
                pt2 = pt2->n;
            }
        }

        // Pre-asignar la lista completa de una sola vez
        out = PyList_New(total_size);
        long idx = 0;
        
        for( i = 0; i < cpu; i++ ) {
            two_lst *pt2 = thread_nbn[i]->n;
            while( pt2 != NULL ) {
                PyObject *sub = PyList_New(3);
                PyList_SET_ITEM(sub, 0, PyLong_FromLong(pt2->i));
                PyList_SET_ITEM(sub, 1, PyLong_FromLong(pt2->j));
                PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(1.0));
                
                PyList_SET_ITEM(out, idx++, sub);

                two_lst *tmp = pt2;
                pt2 = pt2->n;
                free(tmp);
            }
            free( thread_nbn[i] );
        }
        free( thread_nbn );

        for( i = 0; i < n_dih; i++ ) {
            PyObject *sub = PyList_New(3);
            PyList_SET_ITEM(sub, 0, PyLong_FromLong(dih[2*i]));
            PyList_SET_ITEM(sub, 1, PyLong_FromLong(dih[2*i+1]));
            PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(0.5));
            PyList_SET_ITEM(out, idx++, sub);
        }

        free( fre ); free( qms ); free( bnd ); free( ang ); free( dih ); free( xyz );
        return( out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_bond( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *grd, *itm, *dat;
    long            *siz, i, j, k, cpu;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        otmp = PyObject_GetAttrString( object, "ncpu" );
        cpu  = PyLong_AsLong( otmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = ( Py_True == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        if( gradient != Py_True ) { grd = NULL; }
        else {
            grd = (double*) malloc( cpu * siz[0] * siz[1] * sizeof( double ) );
            for( i = 0; i < cpu * siz[0] * siz[1]; i++ ) grd[i] = 0.0;
        }

        otmp  = PyObject_GetAttrString( object, "bond" );
        n_lst = PyList_Size( otmp );
        lst   = (long*) malloc( 2*n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 2; j++ ) lst[2*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp  = PyObject_GetAttrString( object, "bond_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( 2*n_dat * sizeof( double ) );
        for( i = 0; i < n_dat; i++ ) 
            for( j = 0; j < 2; j++ ) dat[2*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "bond_indx" );
        ind    = (long*) malloc( n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        omp_set_num_threads(cpu);
        #pragma omp parallel reduction(+:out)
        {
            int tid = omp_get_thread_num();
            long who = tid * siz[0] * siz[1];
            long _i, _j, ai, aj;
            double vec[3], val, dif, tmp;

            #pragma omp for schedule(static)
            for( _i = 0; _i < n_lst; _i++ ) {
                if( fre[lst[2*_i]] || fre[lst[2*_i+1]] ) {
                    ai = 3 * lst[2*_i];
                    aj = 3 * lst[2*_i+1];
                    for( _j = 0; _j < 3; _j++ ) vec[_j] = xyz[ai+_j] - xyz[aj+_j];
                    val = sqrt( vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] );
                    dif = val - dat[ind[_i]*2+1];
                    tmp = dif * dat[ind[_i]*2];
                    out += 0.5 * tmp * dif;
                    if( grd != NULL ) {
                        tmp *= 1.0 / val;
                        for( _j = 0; _j < 3; _j++ ) {
                            grd[who+ai+_j] += tmp * vec[_j];
                            grd[who+aj+_j] -= tmp * vec[_j];
                        }
                    }
                }
            }
        }

        if( grd != NULL ) {
            otmp = PyObject_GetAttrString( molecule, "grad" );
            mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
            long num_atoms = siz[0];
            
            #pragma omp parallel for schedule(static)
            for( i = 0; i < num_atoms; i++ ) {
                double *itm_x = (double*) PyArray_GETPTR2( mtmp, i, 0 );
                double *itm_y = (double*) PyArray_GETPTR2( mtmp, i, 1 );
                double *itm_z = (double*) PyArray_GETPTR2( mtmp, i, 2 );
                double gx = 0.0, gy = 0.0, gz = 0.0;
                long offset;
                for( long _k = 0; _k < cpu; _k++ ) {
                    offset = num_atoms * 3 * _k + 3 * i;
                    gx += grd[offset + 0];
                    gy += grd[offset + 1];
                    gz += grd[offset + 2];
                }
                *itm_x += gx;
                *itm_y += gy;
                *itm_z += gz;
            }
            Py_DECREF( mtmp );
            Py_DECREF( otmp );
        }

        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre ); free( grd );
        return( Py_BuildValue( "d", out ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_angle( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *grd, *itm, *dat;
    long            *siz, i, j, k, cpu;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        otmp = PyObject_GetAttrString( object, "ncpu" );
        cpu  = PyLong_AsLong( otmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = ( Py_True == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        if( gradient != Py_True ) { grd = NULL; }
        else {
            grd = (double*) malloc( cpu * siz[0] * siz[1] * sizeof( double ) );
            for( i = 0; i < cpu * siz[0] * siz[1]; i++ ) grd[i] = 0.0;
        }

        otmp  = PyObject_GetAttrString( object, "angl" );
        n_lst = PyList_Size( otmp );
        lst   = (long*) malloc( 3*n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 3; j++ ) lst[3*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp  = PyObject_GetAttrString( object, "angl_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( 2*n_dat * sizeof( double ) );
        for( i = 0; i < n_dat; i++ ) 
            for( j = 0; j < 2; j++ ) dat[2*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "angl_indx" );
        ind    = (long*) malloc( n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        omp_set_num_threads(cpu);
        #pragma omp parallel reduction(+:out)
        {
            int tid = omp_get_thread_num();
            long who = tid * siz[0] * siz[1];
            long _i, _j, ai, aj, ak;
            double dij[3], rij, dkj[3], rkj, val, dif, tmp, fac, dti[3], dtj[3], dtk[3];

            #pragma omp for schedule(static)
            for( _i = 0; _i < n_lst; _i++ ) {
                if( fre[lst[3*_i]] || fre[lst[3*_i+1]] || fre[lst[3*_i+2]] ) {
                    ai = 3 * lst[3*_i];
                    aj = 3 * lst[3*_i+1];
                    ak = 3 * lst[3*_i+2];
                    for( _j = 0; _j < 3; _j++ ) dij[_j] = xyz[ai+_j] - xyz[aj+_j];
                    rij = sqrt( dij[0]*dij[0] + dij[1]*dij[1] + dij[2]*dij[2] );
                    for( _j = 0; _j < 3; _j++ ) dij[_j] /= rij;
                    for( _j = 0; _j < 3; _j++ ) dkj[_j] = xyz[ak+_j] - xyz[aj+_j];
                    rkj = sqrt( dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2] );
                    for( _j = 0; _j < 3; _j++ ) dkj[_j] /= rkj;
                    for( fac = 0.0, _j = 0; _j < 3; _j++ ) fac += dij[_j] * dkj[_j];
                    fac = min( fabs( fac ), 1.0 - 1.0e-6 ) * fac / fabs( fac );
                    val = acos( fac );
                    dif = val - dat[ind[_i]*2+1];
                    tmp = dif * dat[ind[_i]*2];
                    out += 0.5 * tmp * dif;
                    if( grd != NULL ) {
                        tmp *= -1.0 / sqrt( 1.0 - fac * fac );
                        for( _j = 0; _j < 3; _j++ ) {
                            dti[_j] = ( dkj[_j] - fac * dij[_j] ) / rij;
                            dtk[_j] = ( dij[_j] - fac * dkj[_j] ) / rkj;
                            dtj[_j] = - ( dti[_j] + dtk[_j] );
                        }
                        for( _j = 0; _j < 3; _j++ ) {
                            grd[who+ai+_j] += tmp * dti[_j];
                            grd[who+aj+_j] += tmp * dtj[_j];
                            grd[who+ak+_j] += tmp * dtk[_j];
                        }
                    }
                }
            }
        }

        if( grd != NULL ) {
            otmp = PyObject_GetAttrString( molecule, "grad" );
            mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
            long num_atoms = siz[0];
            
            #pragma omp parallel for schedule(static)
            for( i = 0; i < num_atoms; i++ ) {
                double *itm_x = (double*) PyArray_GETPTR2( mtmp, i, 0 );
                double *itm_y = (double*) PyArray_GETPTR2( mtmp, i, 1 );
                double *itm_z = (double*) PyArray_GETPTR2( mtmp, i, 2 );
                double gx = 0.0, gy = 0.0, gz = 0.0;
                long offset;
                for( long _k = 0; _k < cpu; _k++ ) {
                    offset = num_atoms * 3 * _k + 3 * i;
                    gx += grd[offset + 0];
                    gy += grd[offset + 1];
                    gz += grd[offset + 2];
                }
                *itm_x += gx;
                *itm_y += gy;
                *itm_z += gz;
            }
            Py_DECREF( mtmp );
            Py_DECREF( otmp );
        }

        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre ); free( grd );
        return( Py_BuildValue( "d", out ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_dihedral( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *grd, *itm, *dat;
    long            *siz, i, j, k, cpu;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        otmp = PyObject_GetAttrString( object, "ncpu" );
        cpu  = PyLong_AsLong( otmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = ( Py_True == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        if( gradient != Py_True ) { grd = NULL; }
        else {
            grd = (double*) malloc( cpu * siz[0] * siz[1] * sizeof( double ) );
            for( i = 0; i < cpu * siz[0] * siz[1]; i++ ) grd[i] = 0.0;
        }

        otmp  = PyObject_GetAttrString( object, "dihe" );
        n_lst = PyList_Size( otmp );
        lst   = (long*) malloc( 4*n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 4; j++ ) lst[4*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
    
        otmp  = PyObject_GetAttrString( object, "dihe_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( 12*n_dat * sizeof( double ) );
        for( i = 0; i < 12*n_dat; i++ ) dat[i] = 0.0;
        for( i = 0; i < n_dat; i++ )
            for( j = 0; j < 12; j++ ) dat[12*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "dihe_indx" );
        ind    = (long*) malloc( n_lst * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        omp_set_num_threads(cpu);
        #pragma omp parallel reduction(+:out)
        {
            int tid = omp_get_thread_num();
            long who = tid * siz[0] * siz[1];
            long _i, _j, ai, aj, ak, al;
            double rkj, rt2, ru2, rtu, cd, sd, dph;
            double dji[3], dkj[3], dlk[3], vt[3], vu[3], vtu[3], dki[3], dlj[3], dvt[3], dvu[3];
            double cs1, cs2, cs3, cs4, cs5, cs6;
            double sn1, sn2, sn3, sn4, sn5, sn6;

            #pragma omp for schedule(static)
            for( _i = 0; _i < n_lst; _i++ ) {
                if( fre[lst[4*_i]] || fre[lst[4*_i+1]] || fre[lst[4*_i+2]] || fre[lst[4*_i+3]] ) {
                    ai = 3 * lst[4*_i];
                    aj = 3 * lst[4*_i+1];
                    ak = 3 * lst[4*_i+2];
                    al = 3 * lst[4*_i+3];
                    for( _j = 0; _j < 3; _j++ ) dji[_j] = xyz[aj+_j] - xyz[ai+_j];
                    for( _j = 0; _j < 3; _j++ ) dkj[_j] = xyz[ak+_j] - xyz[aj+_j];
                    rkj = sqrt( dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2] );
                    for( _j = 0; _j < 3; _j++ ) dlk[_j] = xyz[al+_j] - xyz[ak+_j];
                    vt[0] = dji[1] * dkj[2] - dkj[1] * dji[2];
                    vt[1] = dji[2] * dkj[0] - dkj[2] * dji[0];
                    vt[2] = dji[0] * dkj[1] - dkj[0] * dji[1];
                    rt2 = vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2];
                    vu[0] = dkj[1] * dlk[2] - dlk[1] * dkj[2];
                    vu[1] = dkj[2] * dlk[0] - dlk[2] * dkj[0];
                    vu[2] = dkj[0] * dlk[1] - dlk[0] * dkj[1];
                    ru2 = vu[0]*vu[0] + vu[1]*vu[1] + vu[2]*vu[2];
                    vtu[0] = vt[1] * vu[2] - vu[1] * vt[2];
                    vtu[1] = vt[2] * vu[0] - vu[2] * vt[0];
                    vtu[2] = vt[0] * vu[1] - vu[0] * vt[1];
                    rtu = sqrt( rt2 * ru2 );
                    if( rtu == 0.0 ) { continue; }
                    for( cs1 = 0.0, _j = 0; _j < 3; _j++ ) cs1 += vt[_j] * vu[_j]; cs1 /= rtu;
                    for( sn1 = 0.0, _j = 0; _j < 3; _j++ ) sn1 += dkj[_j] * vtu[_j]; sn1 /= ( rtu * rkj );
                    cs2 = cs1 * cs1 - sn1 * sn1;
                    sn2 = 2.0 * cs1 * sn1;
                    cs3 = cs1 * cs2 - sn1 * sn2;
                    sn3 = cs1 * sn2 + sn1 * cs2;
                    cs4 = cs1 * cs3 - sn1 * sn3;
                    sn4 = cs1 * sn3 + sn1 * cs3;
                    cs5 = cs1 * cs4 - sn1 * sn4;
                    sn5 = cs1 * sn4 + sn1 * cs4;
                    cs6 = cs1 * cs5 - sn1 * sn5;
                    sn6 = cs1 * sn5 + sn1 * cs5;
                    dph = 0.0;
                    if( dat[12*ind[_i]] != 0.0 ) { 
                        cd    = cos( dat[12*ind[_i]+1] );
                        sd    = sin( dat[12*ind[_i]+1] );
                        dph  += dat[12*ind[_i]] * ( cs1 * sd - sn1 * cd );
                        out  += dat[12*ind[_i]] * ( 1.0 + cs1 * cd + sn1 * sd );
                    }
                    if( dat[12*ind[_i]+2] != 0.0 ) { 
                        cd    = cos( dat[12*ind[_i]+3] );
                        sd    = sin( dat[12*ind[_i]+3] );
                        dph  += dat[12*ind[_i]+2] * 2.0 * ( cs2 * sd - sn2 * cd );
                        out  += dat[12*ind[_i]+2] * ( 1.0 + cs2 * cd + sn2 * sd );
                    }
                    if( dat[12*ind[_i]+4] != 0.0 ) { 
                        cd    = cos( dat[12*ind[_i]+5] );
                        sd    = sin( dat[12*ind[_i]+5] );
                        dph  += dat[12*ind[_i]+4] * 3.0 * ( cs3 * sd - sn3 * cd );
                        out  += dat[12*ind[_i]+4] * ( 1.0 + cs3 * cd + sn3 * sd );
                    }
                    if( dat[12*ind[_i]+6] != 0.0) { 
                        cd    = cos( dat[12*ind[_i]+7] );
                        sd    = sin( dat[12*ind[_i]+7] );
                        dph  += dat[12*ind[_i]+6] * 4.0 * ( cs4 * sd - sn4 * cd );
                        out  += dat[12*ind[_i]+6] * ( 1.0 + cs4 * cd + sn4 * sd );
                    }
                    if( dat[12*ind[_i]+8] != 0.0 ) { 
                        cd    = cos( dat[12*ind[_i]+9] );
                        sd    = sin( dat[12*ind[_i]+9] );
                        dph  += dat[12*ind[_i]+8] * 5.0 * ( cs5 * sd - sn5 * cd );
                        out  += dat[12*ind[_i]+8] * ( 1.0 + cs5 * cd + sn5 * sd );
                    }
                    if( dat[12*ind[_i]+10] != 0.0 ) { 
                        cd    = cos( dat[12*ind[_i]+11] );
                        sd    = sin( dat[12*ind[_i]+11] );
                        dph  += dat[12*ind[_i]+10] * 6.0 * ( cs6 * sd - sn6 * cd );
                        out  += dat[12*ind[_i]+10] * ( 1.0 + cs6 * cd + sn6 * sd );
                    }
                    if( grd != NULL ) {
                        for( _j = 0; _j < 3; _j++ ) dki[_j] = xyz[ak+_j] - xyz[ai+_j];
                        for( _j = 0; _j < 3; _j++ ) dlj[_j] = xyz[al+_j] - xyz[aj+_j];
                        dvt[0] = ( vt[1] * dkj[2] - dkj[1] * vt[2] ) / ( rt2 * rkj );
                        dvt[1] = ( vt[2] * dkj[0] - dkj[2] * vt[0] ) / ( rt2 * rkj );
                        dvt[2] = ( vt[0] * dkj[1] - dkj[0] * vt[1] ) / ( rt2 * rkj );
                        dvu[0] = ( vu[1] * dkj[2] - dkj[1] * vu[2] ) / ( ru2 * rkj );
                        dvu[1] = ( vu[2] * dkj[0] - dkj[2] * vu[0] ) / ( ru2 * rkj );
                        dvu[2] = ( vu[0] * dkj[1] - dkj[0] * vu[1] ) / ( ru2 * rkj );
                        grd[who+ai]   += ( dkj[2] * dvt[1] - dkj[1] * dvt[2] ) * dph;
                        grd[who+ai+1] += ( dkj[0] * dvt[2] - dkj[2] * dvt[0] ) * dph;
                        grd[who+ai+2] += ( dkj[1] * dvt[0] - dkj[0] * dvt[1] ) * dph;
                        grd[who+aj]   += ( dki[1] * dvt[2] - dki[2] * dvt[1] - dlk[2] * dvu[1] + dlk[1] * dvu[2] ) * dph;
                        grd[who+aj+1] += ( dki[2] * dvt[0] - dki[0] * dvt[2] - dlk[0] * dvu[2] + dlk[2] * dvu[0] ) * dph;
                        grd[who+aj+2] += ( dki[0] * dvt[1] - dki[1] * dvt[0] - dlk[1] * dvu[0] + dlk[0] * dvu[1] ) * dph;
                        grd[who+ak]   += ( dji[2] * dvt[1] - dji[1] * dvt[2] - dlj[1] * dvu[2] + dlj[2] * dvu[1] ) * dph;
                        grd[who+ak+1] += ( dji[0] * dvt[2] - dji[2] * dvt[0] - dlj[2] * dvu[0] + dlj[0] * dvu[2] ) * dph;
                        grd[who+ak+2] += ( dji[1] * dvt[0] - dji[0] * dvt[1] - dlj[0] * dvu[1] + dlj[1] * dvu[0] ) * dph;
                        grd[who+al]   += ( - dkj[2] * dvu[1] + dkj[1] * dvu[2] ) * dph;
                        grd[who+al+1] += ( - dkj[0] * dvu[2] + dkj[2] * dvu[0] ) * dph;
                        grd[who+al+2] += ( - dkj[1] * dvu[0] + dkj[0] * dvu[1] ) * dph;
                    }
                }
            }
        }
    
        if( grd != NULL ) {
            otmp = PyObject_GetAttrString( molecule, "grad" );
            mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
            long num_atoms = siz[0];
            
            #pragma omp parallel for schedule(static)
            for( i = 0; i < num_atoms; i++ ) {
                double *itm_x = (double*) PyArray_GETPTR2( mtmp, i, 0 );
                double *itm_y = (double*) PyArray_GETPTR2( mtmp, i, 1 );
                double *itm_z = (double*) PyArray_GETPTR2( mtmp, i, 2 );
                double gx = 0.0, gy = 0.0, gz = 0.0;
                long offset;
                for( long _k = 0; _k < cpu; _k++ ) {
                    offset = num_atoms * 3 * _k + 3 * i;
                    gx += grd[offset + 0];
                    gy += grd[offset + 1];
                    gz += grd[offset + 2];
                }
                *itm_x += gx;
                *itm_y += gy;
                *itm_z += gz;
            }
            Py_DECREF( mtmp );
            Py_DECREF( otmp );
        }

        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre ); free( grd );
        return( Py_BuildValue( "d", out ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_non_bonded( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *grd, *xyz, *dat, *scl, *itm, *qms;
    long            *siz, i, j, k, cpu;
    long            *lst, n_lst;
    double          oel = 0.0, olj = 0.0, con, cof, box[3], epsi, epsf;

    if( PyArg_ParseTuple( args, "OOOd", &object, &molecule, &gradient, &epsi ) ) {
        otmp = PyObject_GetAttrString( object, "ncpu" );
        cpu  = PyLong_AsLong( otmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "boxl" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < 3; i++ ) {
            itm = (double*) PyArray_GETPTR1( mtmp, i );
            box[i] = *itm;
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) {
                itm = (double*) PyArray_GETPTR2( mtmp, i, j );
                xyz[k++] = *itm;
            }
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "qmat" );
        qms  = (double*) malloc( siz[0] * sizeof( double ) );
        for( i = 0; i < siz[0]; i++ ) qms[i] = (double) ( Py_False == PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        dat  = (double*) malloc( 3 * siz[0] * sizeof( double ) );
        otmp = PyObject_GetAttrString( molecule, "epsi" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) {
            itm = (double*) PyArray_GETPTR1( mtmp, i );
            dat[3*i] = *itm;
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );
        
        otmp = PyObject_GetAttrString( molecule, "rmin" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) {
            itm = (double*) PyArray_GETPTR1( mtmp, i );
            dat[3*i+1] = *itm;
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );
        
        otmp = PyObject_GetAttrString( molecule, "chrg" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) {
            itm = (double*) PyArray_GETPTR1( mtmp, i );
            dat[3*i+2] = *itm;
        }
        Py_DECREF( mtmp );
        Py_DECREF( otmp );

        if( gradient != Py_True ) { grd = NULL; }
        else {
            grd = (double*) malloc( cpu * siz[0] * siz[1] * sizeof( double ) );
            for( i = 0; i < cpu * siz[0] * siz[1]; i++ ) grd[i] = 0.0;
        }

        otmp = PyObject_GetAttrString( object, "cut_on" );
        con = PyFloat_AsDouble( otmp );
        Py_DECREF( otmp );
    
        otmp = PyObject_GetAttrString( object, "cut_off" );
        cof = PyFloat_AsDouble( otmp );
        Py_DECREF( otmp );
    
        otmp  = PyObject_GetAttrString( object, "nbnd" );
        n_lst = PyList_Size( otmp );
        lst   = (long*) malloc( 2*n_lst * sizeof( long ) );
        scl   = (double*) malloc( n_lst * sizeof( double ) );
        for( i = 0; i < n_lst; i++ ) {
            lst[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            lst[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 1 ) );
            scl[i]     = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), 2 ) );
        }
        Py_DECREF( otmp );

        epsf = 1389.35484620709144110151 / epsi;
        omp_set_num_threads(cpu);

        if( con > 0.0 && cof > con ) {
            double c2on = con * con;
            double c2of = cof * cof;
            double _g   = pow( c2of - c2on, 3.0 );
            double _a   = c2of * c2of * ( c2of - 3.0 * c2on ) / _g;
            double _b   = 6.0 * c2of * c2on / _g;
            double _c   = - ( c2of + c2on ) / _g;
            double _d   = 0.4 / _g;
            double _el1 = 8.0 * ( c2of * c2on * ( cof - con ) - 0.2 * ( cof * c2of * c2of - con * c2on * c2on ) ) / _g;
            double _el2 = - _a / cof + _b * cof + _c * cof * c2of + _d * cof * c2of * c2of;
            double k6   = ( cof * c2of ) / ( cof * c2of - con * c2on );
            double k12  = pow( c2of, 3.0 ) / ( pow( c2of, 3.0 ) - pow( c2on, 3.0 ) );

            #pragma omp parallel reduction(+:oel, olj)
            {
                int tid = omp_get_thread_num();
                long who = tid * siz[0] * siz[1];
                long _i, _j, ii, jj, ai, aj;
                double dr[3], r2, eij, sij, qij, r, s6, tmp, df;
                double _lj1, _lj2, r3, r5, s, s3, s12;

                #pragma omp for schedule(guided)
                for( _i = 0; _i < n_lst; _i++ ) {
                    ii = lst[2*_i];
                    jj = lst[2*_i+1];
                    ai = 3 * ii;
                    aj = 3 * jj;
                    for( _j = 0; _j < 3; _j++ ) dr[_j] = xyz[ai+_j] - xyz[aj+_j];
                    for( _j = 0; _j < 3; _j++ ) dr[_j] -= box[_j] * round( dr[_j] / box[_j] );
                    r2  = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                    if( r2 > c2of ) { continue; }
                    eij = dat[3*ii] * dat[3*jj];
                    sij = dat[3*ii+1] + dat[3*jj+1];
                    qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];
                    r   = sqrt( r2 );
                    s   = 1.0 / r;
                    s3  = pow( sij * s, 3.0 );
                    s6  = s3 * s3;
                    
                    if( r2 <= c2on ) {
                        tmp = qij * s;
                        oel += scl[_i] * ( tmp + qij * _el1 );
                        s12  = s6 * s6;
                        _lj1 = pow( sij / cof * sij / con, 3.0 );
                        _lj2 = _lj1 * _lj1;
                        olj += scl[_i] * eij * ( ( s12 - _lj2 ) - 2.0 * ( s6 - _lj1 ) );
                        df   = ( 12.0 * eij * ( s6 - s12 ) - tmp ) / r2;
                    } else {
                        r3   = r * r2;
                        r5   = r3 * r2;
                        oel += scl[_i] * qij * ( _a * s - _b * r - _c * r3 - _d * r5 + _el2 );
                        _lj1 = pow( sij / cof, 3.0 );
                        _lj2 = _lj1 * _lj1;
                        olj += scl[_i] * eij * ( k12 * pow( s6 - _lj2, 2.0 ) - 2.0 * k6 * pow( s3 - _lj1, 2.0 ) );
                        df   = - qij * ( _a / r3 + _b * s + 3.0 * _c * r + 5.0 * _d * r3 ) ;
                        df  -= 12.0 * eij * ( k12 * s6 * ( s6 - _lj2 ) - k6 * s3 * ( s3 - _lj1 ) ) / r2;
                    }
                    if( grd != NULL ) {
                        for( _j = 0; _j < 3; _j++ ) {
                            grd[who+ai+_j] += scl[_i] * df * dr[_j];
                            grd[who+aj+_j] -= scl[_i] * df * dr[_j];
                        }
                    }
                }
            }

        } else {
            #pragma omp parallel reduction(+:oel, olj)
            {
                int tid = omp_get_thread_num();
                long who = tid * siz[0] * siz[1];
                long _i, _j, ii, jj, ai, aj;
                double dr[3], r2, eij, sij, qij, s6, tmp, df, s;

                #pragma omp for schedule(guided)
                for( _i = 0; _i < n_lst; _i++ ) {
                    ii = lst[2*_i];
                    jj = lst[2*_i+1];
                    ai = 3 * ii;
                    aj = 3 * jj;
                    for( _j = 0; _j < 3; _j++ ) dr[_j] = xyz[ai+_j] - xyz[aj+_j];
                    for( _j = 0; _j < 3; _j++ ) dr[_j] -= box[_j] * round( dr[_j] / box[_j] );
                    r2  = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                    eij = dat[3*ii] * dat[3*jj];
                    sij = dat[3*ii+1] + dat[3*jj+1];
                    qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];
                    s   = 1.0 / sqrt( r2 );
                    s6  = pow( sij * s, 6.0 );
                    tmp = qij * s;
                    oel += scl[_i] * tmp;
                    olj += scl[_i] * eij * s6 * ( s6 - 2.0 );
                    
                    if( grd != NULL ) {
                        df = scl[_i] * ( 12.0 * eij * s6 * ( 1.0 - s6 ) - tmp ) / r2;
                        for( _j = 0; _j < 3; _j++ ) {
                            grd[who+ai+_j] += df * dr[_j];
                            grd[who+aj+_j] -= df * dr[_j];
                        }
                    }
                }
            }
        }
    
        if( grd != NULL ) {
            otmp = PyObject_GetAttrString( molecule, "grad" );
            mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
            long num_atoms = siz[0];
            
            #pragma omp parallel for schedule(static)
            for( i = 0; i < num_atoms; i++ ) {
                double *itm_x = (double*) PyArray_GETPTR2( mtmp, i, 0 );
                double *itm_y = (double*) PyArray_GETPTR2( mtmp, i, 1 );
                double *itm_z = (double*) PyArray_GETPTR2( mtmp, i, 2 );
                double gx = 0.0, gy = 0.0, gz = 0.0;
                long offset;
                for( long _k = 0; _k < cpu; _k++ ) {
                    offset = num_atoms * 3 * _k + 3 * i;
                    gx += grd[offset + 0];
                    gy += grd[offset + 1];
                    gz += grd[offset + 2];
                }
                *itm_x += gx;
                *itm_y += gy;
                *itm_z += gz;
            }
            Py_DECREF( mtmp );
            Py_DECREF( otmp );
        }

        free( xyz ); free( dat ); free ( lst ); free( scl ); free( qms ); free( grd );
        return( Py_BuildValue( "(d,d)", oel, olj ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static struct PyMethodDef methods [] = {
    { "guess_angles",      (PyCFunction)w_guess_angles,      METH_VARARGS },
    { "guess_dihedrals",   (PyCFunction)w_guess_dihedrals,   METH_VARARGS },
    { "update_non_bonded", (PyCFunction)w_update_non_bonded, METH_VARARGS },
    { "ebond",             (PyCFunction)w_energy_bond,       METH_VARARGS },
    { "eangle",            (PyCFunction)w_energy_angle,      METH_VARARGS },
    { "edihedral",         (PyCFunction)w_energy_dihedral,   METH_VARARGS },
    { "enonbonded",        (PyCFunction)w_energy_non_bonded, METH_VARARGS },
    { 0, 0, 0 }
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_molmech",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__molmech( void ) {
    PyObject    *my_module;
    my_module = PyModule_Create( &moddef );
    import_array();
    return( my_module );
}
