#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <pthread.h>


/*
 * Sample of engine using a neuronal network model trained with the correction energies
 *

class mlcor( object ):
    def __init__( self, mol ):
        with open( "data.pk", "rb" ) as f:
            inp   = pickle.load( f )
            l_ene = pickle.load( f )
            h_ene = pickle.load( f )
        out        = h_ene - l_ene
        self.e_min = float( numpy.min( out, axis = 0 ) )
        self.e_dsp = float( numpy.max( out, axis = 0 ) ) - self.e_min
        self.ener  = tf.keras.models.load_model( "ener.h5" )
        self.ones  = numpy.ones( mol.natm )

    def get_grad( self, mol ):
        inp = qm3.engines._ml_info.coul_info( self.ones, mol.coor )
#        inp = qm3.engines._ml_info.acsf_info( 4.0, [1.0], 1.0, [0.1], mol.coor )
        inp = tf.convert_to_tensor( inp.reshape( ( 1, len( inp ) ) ) )
        # ------------------------------------------------
        with tf.GradientTape() as grd:
            grd.watch( inp )
            lss = self.ener( inp )
        ene = float( self.ener( inp, training = False ) )
        grd = grd.gradient( lss, inp ).numpy().ravel()
        # ------------------------------------------------
        mol.func += ene * self.e_dsp + self.e_min
        grd = numpy.dot( grd.T, qm3.engines._ml_info.coul_jaco( self.ones, mol.coor ) )
#        grd = numpy.dot( grd.T, qm3.engines._ml_info.acsf_jaco( 4.0, [1.0], 1.0, [0.1], mol.coor ) )
        grd.shape = ( mol.natm, 3 )
        mol.grad += grd * self.e_dsp

*/


double __dist( long i3, long j3, double *xyz ) {
    return( sqrt(
            ( xyz[j3]   - xyz[i3]   ) * ( xyz[j3]   - xyz[i3]   ) +
            ( xyz[j3+1] - xyz[i3+1] ) * ( xyz[j3+1] - xyz[i3+1] ) +
            ( xyz[j3+2] - xyz[i3+2] ) * ( xyz[j3+2] - xyz[i3+2] ) ) );
}


static PyObject* _Coul_info( PyObject *self, PyObject *args ) {
    PyObject		*ocrd, *onum;
	PyArrayObject	*mcrd, *mnum, *out;
	long			*siz, dim[1], i, j, k, ww;
	double			*num, *xyz, *itm;

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
		for( i = 0; i < siz[0]; i++ ) {
			k    = i * 3;
			ww   = i * siz[0] - ( ( i - 1 ) * i ) / 2;
			itm  = (double*) PyArray_GETPTR1( out, ww );
			*itm = 0.5 * pow( num[i], 2.4 );
			for( j = i + 1; j < siz[0]; j++ ) {
				itm  = (double*) PyArray_GETPTR1( out, ww+j-i );
				*itm = num[i] * num[j] / __dist( k, j * 3, xyz );
			}
		}
		free( num ); free( xyz );
		return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static PyObject* _Coul_jaco( PyObject *self, PyObject *args ) {
    PyObject		*ocrd, *onum;
	PyArrayObject	*mcrd, *mnum, *out;
	long			*siz, dim[2], i, j, k, i3, j3, row;
	double			*num, *xyz, dr[3], r2, zz, *itm;

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
        Py_DECREF( mcrd );
        Py_DECREF( mnum );
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
        Py_DECREF( mcrd );
        Py_DECREF( mnum );
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
                zz = num[i] * num[j] / ( r2 * sqrt( r2 ) );
                for( k = 0; k < 3; k ++ ) {
                    itm  = (double*) PyArray_GETPTR2( out, row, i3+k );
                    *itm =   dr[k] * zz;
                    itm  = (double*) PyArray_GETPTR2( out, row, j3+k );
                    *itm = - dr[k] * zz;
                }
                row++;
            }
        }
        free( num ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


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


/* [doi:10.1063/1.3553717]
f( R_{i,j} ) = 1 over 2 left( 1 + cos left( %pi R_{i,j} over R_c right) right )
newline
g^2_{ i,j,l } = sum from{ j <> i } e^{- %eta_l R_{i,j}^2 } · f( R_{i,j} )
~~~~
G^2_{ i,l } = sum from{ j <> i } g^2_{ i,j,l }
newline
g^5_{ i,j,k,l } = left(1 + cos %zeta_{i,j,k} right)^%dseda e^{-%ji _l left( R_{i,j}^2 + R_{i,k}^2 right) } · f( R_{i,j} ) · f( R_{i,k} )
~~~~
G^5_{ i,l } = 2^{1-%dseda } sum from{ j <> i, k <> i,j } g^5_{ i,j,k,l }
newline
3N rightarrow N(n_2 + n_5): [ G^2_{1,1} ~ dotsaxis ~ G^2_{1,n_2} ~ G^5_{1,1} ~ dotsaxis ~ G^5_{1,n_5} ~ dotsaxis ~ dotsaxis ~ G^2_{N,1} ~ dotsaxis ~ G^2_{N,n_2} ~ G^5_{N,1} ~ dotsaxis ~ G^5_{N,n_5} ]
*/


static PyObject* _acsf_info( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *oeta2, *oeta5;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim, odim[1], i, i3, j, j3, k, k3, l, dd, neta2, neta5;
    double          *xyz, cutx, dse5, pre5, *eta2, *eta5, fij, fik, dij, dik, *itm;

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
        Py_DECREF( mcrd );
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
        free( eta2 ); free( eta5 ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static PyObject* _acsf_jaco( PyObject *self, PyObject *args ) {
    PyObject        *ocrd, *oeta2, *oeta5;
    PyArrayObject   *mcrd, *out;
    long            *siz, dim, odim[2], i, i3, j, j3, k, k3, l, m, dd, neta2, neta5;
    double          *xyz, cutx, dse5, pre5, *eta2, *eta5, rij[3], rik[3], dij2, dik2, fij, fik, dij, dik, *itm, tmp, exp5, pmt;

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
        Py_DECREF( mcrd );
        dim     = neta2 + neta5;
        odim[0] = siz[0] * dim;
        odim[1] = siz[0] * siz[1];
        out     = (PyArrayObject*) PyArray_ZEROS( 2, odim, NPY_DOUBLE, 0 );
        pre5    = pow( 2.0, 1.0 - dse5 );
        for( i = 0; i < siz[0]; i++ ) {
            i3 = i * 3;
            dd = i * dim;
            for( j = 0; j < siz[0]; j++ ) if( j != i ) {
                j3     = j * 3;
                rij[0] = xyz[j3]   - xyz[i3];
                rij[1] = xyz[j3+1] - xyz[i3+1];
                rij[2] = xyz[j3+2] - xyz[i3+2];
                dij2   = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                dij    = sqrt( dij2 );
                if( dij > cutx ) { fij = 0.0; } else { fij = 0.5 * ( cos( M_PI * dij / cutx ) + 1.0 ); }
                if( fij > 0.0 ) {
                    for( l = 0; l < neta2; l++ ) { 
                        tmp = exp( - eta2[l] * dij2 ) * ( 2.0 * fij * eta2[l] + M_PI * sin ( M_PI * dij / cutx ) / ( 2.0 * cutx * dij ) );
                        for( m = 0; m < 3; m++ ) {
                            itm  = (double*) PyArray_GETPTR2( out, dd + l, i3 + m );
                            *itm += tmp * rij[m];
                            itm  = (double*) PyArray_GETPTR2( out, dd + l, j3 + m );
                            *itm -= tmp * rij[m];
                        }
                    }
                    for( k = 0; k < siz[0]; k++ ) if( k != j && k != i ) {
                        k3  = k * 3;
                        rik[0] = xyz[k3]   - xyz[i3];
                        rik[1] = xyz[k3+1] - xyz[i3+1];
                        rik[2] = xyz[k3+2] - xyz[i3+2];
                        dik2   = rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2];
                        dik    = sqrt( dik2 );
                        if( dik > cutx ) { fik = 0.0; } else { fik = 0.5 * ( cos( M_PI * dik / cutx ) + 1.0 ); }
                        if( fik > 0.0 ) {
                            tmp = ( rij[0] * rik[0] + rij[1] * rik[1] + rij[2] * rik[2] ) / ( dij * dik );
                            pmt = pow( 1.0 + tmp, dse5 );
                            for( l = 0; l < neta5; l++ ) {
                                exp5 = exp( - eta5[l] * ( dij2 + dik2 ) );
                                for( m = 0; m < 3; m++ ) {
                                    itm  = (double*) PyArray_GETPTR2( out, dd + neta2 + l, i3 + m );
                                    *itm += 2.0 * pre5 * exp5 * eta5[l] * ( rij[m] + rik[m] ) * pmt * fij * fik;
                                    *itm -= pre5 * dse5 * exp5 * pmt / ( 1.0 + tmp ) * ( ( rij[m] + rik[m] ) / ( dij * dik ) - ( ( dik2 * rij[m] + dij2 * rik[m] ) * tmp ) / ( dij2 * dik2 ) ) * fij * fik;
                                    *itm += 0.5 * pre5 * exp5 * M_PI * pmt * ( rij[m] * fik * sin( M_PI * dij / cutx ) / dij + rik[m] * fij * sin( M_PI * dik / cutx ) / dik ) / cutx;

                                    itm  = (double*) PyArray_GETPTR2( out, dd + neta2 + l, j3 + m );
                                    *itm -= 2.0 * pre5 * exp5 * eta5[l] * rij[m] * pmt * fij * fik;
                                    *itm += pre5 * dse5 * exp5 * pmt / ( 1.0 + tmp ) * ( rik[m] / ( dij * dik ) - dik2 * rij[m] * tmp / ( dij2 * dik2 ) ) * fij * fik;
                                    *itm -= 0.5 * pre5 * exp5 * M_PI * pmt * rij[m] * fik * sin( M_PI * dij / cutx ) / ( cutx * dij );

                                    itm  = (double*) PyArray_GETPTR2( out, dd + neta2 + l, k3 + m );
                                    *itm -= 2.0 * pre5 * exp5 * eta5[l] * rik[m] * pmt * fij * fik;
                                    *itm += pre5 * dse5 * exp5 * pmt / ( 1.0 + tmp ) * ( rij[m] / ( dij * dik ) - dij2 * rik[m] * tmp / ( dij2 * dik2 ) ) * fij * fik;
                                    *itm -= 0.5 * pre5 * exp5 * M_PI * pmt * rik[m] * fij * sin( M_PI * dik / cutx ) / ( cutx * dik );
                                }
                            }
                        }
                    }
                }
            }
        }
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
    double          *xyz, cutx, dse5, pre5, *eta2, *eta5, *itm;
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
        Py_DECREF( mcrd );
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
        for( j = 0; j < siz[0]; j++ ) {
            dd = j * dim;
            for( k = 0; k < neta2; k++ ) {
                itm  = (double*) PyArray_GETPTR1( out, dd + k );
                for( i = 0; i < ncpu; i++ ) *itm += arg[i].out[dd + k];
            }
            for( k = 0; k < neta5; k++ ) {
                itm  = (double*) PyArray_GETPTR1( out, dd + neta2 + k );
                for( i = 0; i < ncpu; i++ ) *itm += arg[i].out[dd + neta2 + k];
            }
        }
        free( arg ); free( pid );
        free( eta2 ); free( eta5 ); free( xyz );
        return( (PyObject*) out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}


static struct PyMethodDef methods [] = {
    { "Coul_info", (PyCFunction)_Coul_info, METH_VARARGS },
    { "Coul_jaco", (PyCFunction)_Coul_jaco, METH_VARARGS },

    { "coul_info", (PyCFunction)_coul_info, METH_VARARGS },
    { "coul_jaco", (PyCFunction)_coul_jaco, METH_VARARGS },
    { "acsf_info", (PyCFunction)_acsf_info, METH_VARARGS },
    { "acsf_pinf", (PyCFunction)_acsf_pinf, METH_VARARGS },
    { "acsf_jaco", (PyCFunction)_acsf_jaco, METH_VARARGS },
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
