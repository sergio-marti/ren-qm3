#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#undef min
#undef max
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/* -------------------------------------------------------------------------
 * Macro de comprobación de errores CUDA
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t _e = (err);                                              \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
        }                                                                    \
    } while(0)

/* Estado global CUDA (equivalente a los cl_context / cl_command_queue) */
static int cuda_initialized = 0;

static void init_cuda(void) {
    if (cuda_initialized) return;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA: %s (CC %d.%d)\n", prop.name,
           prop.major, prop.minor);
    cuda_initialized = 1;
}

/* =========================================================================
 * KERNELS CUDA
 * ====================================================================== */

/* -----------------------------------------------------------------
 * ebond — energía de enlace (harmónico)
 * ---------------------------------------------------------------- */
__global__ void k_ebond(const double* __restrict__ xyz,
                        const long*   __restrict__ lst,
                        const double* __restrict__ dat,
                        const long*   __restrict__ ind,
                        const long*   __restrict__ fre,
                        double* out_eng, double* out_grd,
                        int do_grad, int n_lst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lst) return;

    long ai = lst[2*i] * 3, aj = lst[2*i+1] * 3;

    /* Si ningún átomo está activo, energía = 0 */
    if (!fre[lst[2*i]] && !fre[lst[2*i+1]]) {
        out_eng[i] = 0.0;
        if (do_grad) { for (int j = 0; j < 6; j++) out_grd[i*6+j] = 0.0; }
        return;
    }

    double vec[3], val, dif, tmp;
    vec[0] = xyz[ai]   - xyz[aj];
    vec[1] = xyz[ai+1] - xyz[aj+1];
    vec[2] = xyz[ai+2] - xyz[aj+2];
    val = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    dif = val - dat[ind[i]*2+1];
    tmp = dif * dat[ind[i]*2];
    out_eng[i] = 0.5 * tmp * dif;

    if (do_grad) {
        tmp *= 1.0 / val;
        out_grd[i*6+0] =  tmp * vec[0];
        out_grd[i*6+1] =  tmp * vec[1];
        out_grd[i*6+2] =  tmp * vec[2];
        out_grd[i*6+3] = -tmp * vec[0];
        out_grd[i*6+4] = -tmp * vec[1];
        out_grd[i*6+5] = -tmp * vec[2];
    }
}

/* -----------------------------------------------------------------
 * eangle — energía de ángulo (harmónico)
 * ---------------------------------------------------------------- */
__global__ void k_eangle(const double* __restrict__ xyz,
                         const long*   __restrict__ lst,
                         const double* __restrict__ dat,
                         const long*   __restrict__ ind,
                         const long*   __restrict__ fre,
                         double* out_eng, double* out_grd,
                         int do_grad, int n_lst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lst) return;

    long idx_i = lst[3*i], idx_j = lst[3*i+1], idx_k = lst[3*i+2];
    long ai = idx_i*3, aj = idx_j*3, ak = idx_k*3;

    if (!fre[idx_i] && !fre[idx_j] && !fre[idx_k]) {
        out_eng[i] = 0.0;
        if (do_grad) { for (int j = 0; j < 9; j++) out_grd[i*9+j] = 0.0; }
        return;
    }

    double dij[3], dkj[3], dti[3], dtj[3], dtk[3];

    dij[0] = xyz[ai]   - xyz[aj];
    dij[1] = xyz[ai+1] - xyz[aj+1];
    dij[2] = xyz[ai+2] - xyz[aj+2];
    double rij = sqrt(dij[0]*dij[0] + dij[1]*dij[1] + dij[2]*dij[2]);
    dij[0] /= rij; dij[1] /= rij; dij[2] /= rij;

    dkj[0] = xyz[ak]   - xyz[aj];
    dkj[1] = xyz[ak+1] - xyz[aj+1];
    dkj[2] = xyz[ak+2] - xyz[aj+2];
    double rkj = sqrt(dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2]);
    dkj[0] /= rkj; dkj[1] /= rkj; dkj[2] /= rkj;

    double fac = dij[0]*dkj[0] + dij[1]*dkj[1] + dij[2]*dkj[2];
    double abs_fac = fabs(fac);
    fac = fmin(abs_fac, 1.0 - 1.0e-6) * (fac < 0 ? -1.0 : 1.0);
    double val = acos(fac);
    double dif = val - dat[ind[i]*2+1];
    double tmp = dif * dat[ind[i]*2];
    out_eng[i] = 0.5 * tmp * dif;

    if (do_grad) {
        tmp *= -1.0 / sqrt(1.0 - fac * fac);
        for (int j = 0; j < 3; j++) {
            dti[j] = (dkj[j] - fac * dij[j]) / rij;
            dtk[j] = (dij[j] - fac * dkj[j]) / rkj;
            dtj[j] = -(dti[j] + dtk[j]);
        }
        out_grd[i*9+0] = tmp*dti[0]; out_grd[i*9+1] = tmp*dti[1]; out_grd[i*9+2] = tmp*dti[2];
        out_grd[i*9+3] = tmp*dtj[0]; out_grd[i*9+4] = tmp*dtj[1]; out_grd[i*9+5] = tmp*dtj[2];
        out_grd[i*9+6] = tmp*dtk[0]; out_grd[i*9+7] = tmp*dtk[1]; out_grd[i*9+8] = tmp*dtk[2];
    }
}

/* -----------------------------------------------------------------
 * edihedral — energía diedro (serie de Fourier, hasta 6 términos)
 * ---------------------------------------------------------------- */
__global__ void k_edihedral(const double* __restrict__ xyz,
                            const long*   __restrict__ lst,
                            const double* __restrict__ dat,
                            const long*   __restrict__ ind,
                            const long*   __restrict__ fre,
                            double* out_eng, double* out_grd,
                            int do_grad, int n_lst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lst) return;

    long idx_i = lst[4*i], idx_j = lst[4*i+1],
         idx_k = lst[4*i+2], idx_l = lst[4*i+3];
    long ai = idx_i*3, aj = idx_j*3, ak = idx_k*3, al = idx_l*3;

    if (!fre[idx_i] && !fre[idx_j] && !fre[idx_k] && !fre[idx_l]) {
        out_eng[i] = 0.0;
        if (do_grad) { for (int j = 0; j < 12; j++) out_grd[i*12+j] = 0.0; }
        return;
    }

    double dji[3], dkj[3], dlk[3], vt[3], vu[3], vtu[3];
    double dki[3], dlj[3], dvt[3], dvu[3];

    for (int j = 0; j < 3; j++) {
        dji[j] = xyz[aj+j] - xyz[ai+j];
        dkj[j] = xyz[ak+j] - xyz[aj+j];
        dlk[j] = xyz[al+j] - xyz[ak+j];
    }

    double rkj = sqrt(dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2]);

    vt[0] = dji[1]*dkj[2] - dkj[1]*dji[2];
    vt[1] = dji[2]*dkj[0] - dkj[2]*dji[0];
    vt[2] = dji[0]*dkj[1] - dkj[0]*dji[1];
    double rt2 = vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2];

    vu[0] = dkj[1]*dlk[2] - dlk[1]*dkj[2];
    vu[1] = dkj[2]*dlk[0] - dlk[2]*dkj[0];
    vu[2] = dkj[0]*dlk[1] - dlk[0]*dkj[1];
    double ru2 = vu[0]*vu[0] + vu[1]*vu[1] + vu[2]*vu[2];

    vtu[0] = vt[1]*vu[2] - vu[1]*vt[2];
    vtu[1] = vt[2]*vu[0] - vu[2]*vt[0];
    vtu[2] = vt[0]*vu[1] - vu[0]*vt[1];
    double rtu = sqrt(rt2 * ru2);

    if (rtu == 0.0) {
        out_eng[i] = 0.0;
        if (do_grad) { for (int j = 0; j < 12; j++) out_grd[i*12+j] = 0.0; }
        return;
    }

    double cs1 = 0.0, sn1 = 0.0;
    for (int j = 0; j < 3; j++) { cs1 += vt[j]*vu[j]; sn1 += dkj[j]*vtu[j]; }
    cs1 /= rtu; sn1 /= (rtu * rkj);

    double cs2 = cs1*cs1 - sn1*sn1, sn2 = 2.0*cs1*sn1;
    double cs3 = cs1*cs2 - sn1*sn2, sn3 = cs1*sn2 + sn1*cs2;
    double cs4 = cs1*cs3 - sn1*sn3, sn4 = cs1*sn3 + sn1*cs3;
    double cs5 = cs1*cs4 - sn1*sn4, sn5 = cs1*sn4 + sn1*cs4;
    double cs6 = cs1*cs5 - sn1*sn5, sn6 = cs1*sn5 + sn1*cs5;

    double dph = 0.0, eng = 0.0, cd, sd;
    int idx = ind[i] * 12;
    double c_cs[6] = {cs1, cs2, cs3, cs4, cs5, cs6};
    double c_sn[6] = {sn1, sn2, sn3, sn4, sn5, sn6};

    for (int k = 0; k < 6; k++) {
        if (dat[idx + k*2] != 0.0) {
            cd = cos(dat[idx + k*2 + 1]);
            sd = sin(dat[idx + k*2 + 1]);
            dph += dat[idx+k*2] * (k + 1.0) * (c_cs[k]*sd - c_sn[k]*cd);
            eng += dat[idx+k*2] * (1.0 + c_cs[k]*cd + c_sn[k]*sd);
        }
    }
    out_eng[i] = eng;

    if (do_grad) {
        for (int j = 0; j < 3; j++) {
            dki[j] = xyz[ak+j] - xyz[ai+j];
            dlj[j] = xyz[al+j] - xyz[aj+j];
        }
        dvt[0] = (vt[1]*dkj[2] - dkj[1]*vt[2]) / (rt2*rkj);
        dvt[1] = (vt[2]*dkj[0] - dkj[2]*vt[0]) / (rt2*rkj);
        dvt[2] = (vt[0]*dkj[1] - dkj[0]*vt[1]) / (rt2*rkj);
        dvu[0] = (vu[1]*dkj[2] - dkj[1]*vu[2]) / (ru2*rkj);
        dvu[1] = (vu[2]*dkj[0] - dkj[2]*vu[0]) / (ru2*rkj);
        dvu[2] = (vu[0]*dkj[1] - dkj[0]*vu[1]) / (ru2*rkj);

        out_grd[i*12+ 0] = (dkj[2]*dvt[1] - dkj[1]*dvt[2]) * dph;
        out_grd[i*12+ 1] = (dkj[0]*dvt[2] - dkj[2]*dvt[0]) * dph;
        out_grd[i*12+ 2] = (dkj[1]*dvt[0] - dkj[0]*dvt[1]) * dph;
        out_grd[i*12+ 3] = (dki[1]*dvt[2] - dki[2]*dvt[1] - dlk[2]*dvu[1] + dlk[1]*dvu[2]) * dph;
        out_grd[i*12+ 4] = (dki[2]*dvt[0] - dki[0]*dvt[2] - dlk[0]*dvu[2] + dlk[2]*dvu[0]) * dph;
        out_grd[i*12+ 5] = (dki[0]*dvt[1] - dki[1]*dvt[0] - dlk[1]*dvu[0] + dlk[0]*dvu[1]) * dph;
        out_grd[i*12+ 6] = (dji[2]*dvt[1] - dji[1]*dvt[2] - dlj[1]*dvu[2] + dlj[2]*dvu[1]) * dph;
        out_grd[i*12+ 7] = (dji[0]*dvt[2] - dji[2]*dvt[0] - dlj[2]*dvu[0] + dlj[0]*dvu[2]) * dph;
        out_grd[i*12+ 8] = (dji[1]*dvt[0] - dji[0]*dvt[1] - dlj[0]*dvu[1] + dlj[1]*dvu[0]) * dph;
        out_grd[i*12+ 9] = (-dkj[2]*dvu[1] + dkj[1]*dvu[2]) * dph;
        out_grd[i*12+10] = (-dkj[0]*dvu[2] + dkj[2]*dvu[0]) * dph;
        out_grd[i*12+11] = (-dkj[1]*dvu[0] + dkj[0]*dvu[1]) * dph;
    }
}

/* -----------------------------------------------------------------
 * enonbonded_cut — no enlazados con cutoff suavizado
 * ---------------------------------------------------------------- */
__global__ void k_enonbonded_cut(
        const double* __restrict__ xyz,
        const long*   __restrict__ lst,
        const double* __restrict__ dat,
        const double* __restrict__ qms,
        const double* __restrict__ scl,
        double* out_oel, double* out_olj, double* out_grd,
        double box0, double box1, double box2, double epsf,
        double c2on, double c2of, double con, double cof,
        double _a, double _b, double _c, double _d,
        double _el1, double _el2,
        double k6, double k12,
        int do_grad, int n_lst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lst) return;

    long ii = lst[2*i], jj = lst[2*i+1];
    long ai = 3*ii,     aj = 3*jj;

    double dr[3];
    dr[0] = xyz[ai]   - xyz[aj];
    dr[1] = xyz[ai+1] - xyz[aj+1];
    dr[2] = xyz[ai+2] - xyz[aj+2];
    if (box0 > 0.0) dr[0] -= box0 * round(dr[0]/box0);
    if (box1 > 0.0) dr[1] -= box1 * round(dr[1]/box1);
    if (box2 > 0.0) dr[2] -= box2 * round(dr[2]/box2);

    double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
    if (r2 > c2of) {
        out_oel[i] = 0.0; out_olj[i] = 0.0;
        if (do_grad) { for (int j = 0; j < 6; j++) out_grd[i*6+j] = 0.0; }
        return;
    }

    double eij = dat[3*ii]   * dat[3*jj];
    double sij = dat[3*ii+1] + dat[3*jj+1];
    double qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];

    double r  = sqrt(r2), s = 1.0/r;
    double s3 = (sij*s)*(sij*s)*(sij*s);
    double s6 = s3*s3, s12 = s6*s6;

    double _oel = 0.0, _olj = 0.0, df = 0.0, tmp = 0.0;

    if (r2 <= c2on) {
        tmp  = qij * s;
        _oel = scl[i] * (tmp + qij * _el1);
        double _lj1 = (sij/cof)*(sij/con)*(sij/cof)*(sij/con)*(sij/cof)*(sij/con);
        double _lj2 = _lj1 * _lj1;
        _olj = scl[i] * eij * ((s12 - _lj2) - 2.0 * (s6 - _lj1));
        df   = (12.0 * eij * (s6 - s12) - tmp) / r2;
    } else {
        double r3 = r*r2, r5 = r3*r2;
        _oel = scl[i] * qij * (_a*s - _b*r - _c*r3 - _d*r5 + _el2);
        double _lj1 = (sij/cof)*(sij/cof)*(sij/cof);
        double _lj2 = _lj1 * _lj1;
        _olj = scl[i] * eij * (k12*(s6-_lj2)*(s6-_lj2) - 2.0*k6*(s3-_lj1)*(s3-_lj1));
        df   = -qij * (_a/r3 + _b*s + 3.0*_c*r + 5.0*_d*r3);
        df  -= 12.0 * eij * (k12*s6*(s6-_lj2) - k6*s3*(s3-_lj1)) / r2;
    }

    out_oel[i] = _oel;
    out_olj[i] = _olj;

    if (do_grad) {
        out_grd[i*6+0] =  scl[i]*df*dr[0];
        out_grd[i*6+1] =  scl[i]*df*dr[1];
        out_grd[i*6+2] =  scl[i]*df*dr[2];
        out_grd[i*6+3] = -scl[i]*df*dr[0];
        out_grd[i*6+4] = -scl[i]*df*dr[1];
        out_grd[i*6+5] = -scl[i]*df*dr[2];
    }
}

/* -----------------------------------------------------------------
 * enonbonded_nocut — no enlazados sin cutoff
 * ---------------------------------------------------------------- */
__global__ void k_enonbonded_nocut(
        const double* __restrict__ xyz,
        const long*   __restrict__ lst,
        const double* __restrict__ dat,
        const double* __restrict__ qms,
        const double* __restrict__ scl,
        double* out_oel, double* out_olj, double* out_grd,
        double box0, double box1, double box2, double epsf,
        int do_grad, int n_lst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lst) return;

    long ii = lst[2*i], jj = lst[2*i+1];
    long ai = 3*ii,     aj = 3*jj;

    double dr[3];
    dr[0] = xyz[ai]   - xyz[aj];
    dr[1] = xyz[ai+1] - xyz[aj+1];
    dr[2] = xyz[ai+2] - xyz[aj+2];
    if (box0 > 0.0) dr[0] -= box0 * round(dr[0]/box0);
    if (box1 > 0.0) dr[1] -= box1 * round(dr[1]/box1);
    if (box2 > 0.0) dr[2] -= box2 * round(dr[2]/box2);

    double r2  = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
    double eij = dat[3*ii]   * dat[3*jj];
    double sij = dat[3*ii+1] + dat[3*jj+1];
    double qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];

    double s  = 1.0 / sqrt(r2);
    double s6 = (sij*s)*(sij*s)*(sij*s)*(sij*s)*(sij*s)*(sij*s);
    double tmp = qij * s;

    out_oel[i] = scl[i] * tmp;
    out_olj[i] = scl[i] * eij * s6 * (s6 - 2.0);

    if (do_grad) {
        double df = scl[i] * (12.0 * eij * s6 * (1.0 - s6) - tmp) / r2;
        out_grd[i*6+0] =  df*dr[0];
        out_grd[i*6+1] =  df*dr[1];
        out_grd[i*6+2] =  df*dr[2];
        out_grd[i*6+3] = -df*dr[0];
        out_grd[i*6+4] = -df*dr[1];
        out_grd[i*6+5] = -df*dr[2];
    }
}

/* -----------------------------------------------------------------
 * update_nb — construcción de la lista de pares no enlazados
 *             Grid 2D: cada hilo evalúa el par (i, j)
 * ---------------------------------------------------------------- */
__global__ void k_update_nb(
        const double* __restrict__ xyz,
        const long*   __restrict__ qms,
        const long*   __restrict__ fre,
        const long*   __restrict__ bnd,
        const long*   __restrict__ ang,
        const long*   __restrict__ dih,
        long* out_pairs, int* pair_count,
        int siz, double box0, double box1, double box2, double cut,
        int n_bnd, int n_ang, int n_dih, int max_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= siz || j >= siz || i >= j) return;
    if ((qms[i] == 1 && qms[j] == 1) || (fre[i] == 0 && fre[j] == 0)) return;

    double dr0 = xyz[i*3]   - xyz[j*3];
    double dr1 = xyz[i*3+1] - xyz[j*3+1];
    double dr2 = xyz[i*3+2] - xyz[j*3+2];
    if (box0 > 0.0) dr0 -= box0 * round(dr0/box0);
    if (box1 > 0.0) dr1 -= box1 * round(dr1/box1);
    if (box2 > 0.0) dr2 -= box2 * round(dr2/box2);

    if (dr0*dr0 + dr1*dr1 + dr2*dr2 <= cut) {
        int f = 0;
        for (int k = 0; k < n_bnd && !f; k++)
            f |= ((i==bnd[2*k] && j==bnd[2*k+1]) ||
                  (i==bnd[2*k+1] && j==bnd[2*k]));
        for (int k = 0; k < n_ang && !f; k++)
            f |= ((i==ang[2*k] && j==ang[2*k+1]) ||
                  (i==ang[2*k+1] && j==ang[2*k]));
        for (int k = 0; k < n_dih && !f; k++)
            f |= ((i==dih[2*k] && j==dih[2*k+1]) ||
                  (i==dih[2*k+1] && j==dih[2*k]));
        if (!f) {
            /* atomicAdd devuelve el valor anterior (≡ atomic_inc de OCL) */
            int idx = atomicAdd(pair_count, 1);
            if (idx < max_pairs) {
                out_pairs[2*idx]   = (long)i;
                out_pairs[2*idx+1] = (long)j;
            }
        }
    }
}

/* =========================================================================
 * FUNCIONES AUXILIARES DE MEMORIA
 * ====================================================================== */

/* Alloca memoria en dispositivo y copia datos desde el host */
static void* cuda_malloc_copy(size_t size, const void* host_src) {
    void* d_ptr = NULL;
    CUDA_CHECK(cudaMalloc(&d_ptr, size));
    if (host_src)
        CUDA_CHECK(cudaMemcpy(d_ptr, host_src, size, cudaMemcpyHostToDevice));
    return d_ptr;
}

/* Alloca memoria en dispositivo sin inicializar */
static void* cuda_malloc(size_t size) {
    void* d_ptr = NULL;
    CUDA_CHECK(cudaMalloc(&d_ptr, size));
    return d_ptr;
}

/* Lee desde dispositivo al host (bloqueante) */
static void cuda_read(void* host_dst, const void* d_src, size_t size) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_dst, d_src, size, cudaMemcpyDeviceToHost));
}

/* =========================================================================
 * WRAPPERS PYTHON — sin GPU (igual que en OCL)
 * ====================================================================== */

/* -----------------------------------------------------------------
 * guess_angles — detección de ángulos a partir de la lista de enlaces
 * ---------------------------------------------------------------- */
static PyObject* w_guess_angles(PyObject *self, PyObject *args) {
    PyObject *out, *object, *otmp, *olst;
    long i, j, k, *lst, siz, cnt, ii, jj, kk;
    long nat, **con, *nel;

    if (PyArg_ParseTuple(args, "O", &object)) {
        otmp = PyObject_GetAttrString(object, "bond");
        siz  = PyList_Size(otmp);
        lst  = (long*) malloc(2 * siz * sizeof(long));
        for (i = 0; i < siz; i++)
            for (j = 0; j < 2; j++)
                lst[2*i+j] = PyLong_AsLong(
                    PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        nat  = PyLong_AsLong(PyObject_GetAttrString(object, "natm"));
        otmp = PyObject_GetAttrString(object, "conn");
        con  = (long**) malloc(nat * sizeof(long*));
        nel  = (long*)  malloc(nat * sizeof(long));
        for (i = 0; i < nat; i++) {
            olst    = PyList_GetItem(otmp, i);
            nel[i]  = PyList_Size(olst);
            con[i]  = (long*) malloc(nel[i] * sizeof(long));
            for (j = 0; j < nel[i]; j++)
                con[i][j] = PyLong_AsLong(PyList_GetItem(olst, j));
        }
        Py_DECREF(otmp);

        out = PyList_New(0);
        for (i = 0; i < siz - 1; i++) {
            for (j = i + 1; j < siz; j++) {
                ii = -1; jj = -1; kk = -1;
                if      (lst[2*i]   == lst[2*j])   { ii=lst[2*i+1]; jj=lst[2*i];   kk=lst[2*j+1]; }
                else if (lst[2*i]   == lst[2*j+1]) { ii=lst[2*i+1]; jj=lst[2*i];   kk=lst[2*j];   }
                else if (lst[2*i+1] == lst[2*j])   { ii=lst[2*i];   jj=lst[2*i+1]; kk=lst[2*j+1]; }
                else if (lst[2*i+1] == lst[2*j+1]) { ii=lst[2*i];   jj=lst[2*i+1]; kk=lst[2*j];   }
                if (ii != -1 && jj != -1 && kk != -1) {
                    cnt = 0;
                    for (k = 0; k < nel[ii]; k++) cnt += (kk == con[ii][k]);
                    if (cnt == 0) {
                        otmp = Py_BuildValue("[l,l,l]", ii, jj, kk);
                        PyList_Append(out, otmp);
                        Py_DECREF(otmp);
                    }
                }
            }
        }
        free(lst); free(nel);
        for (i = 0; i < nat; i++) free(con[i]);
        free(con);
        return out;
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* -----------------------------------------------------------------
 * guess_dihedrals — detección de diedros a partir de ángulos
 * ---------------------------------------------------------------- */
static PyObject* w_guess_dihedrals(PyObject *self, PyObject *args) {
    PyObject *out, *object, *otmp, *olst;
    long i, j, k, ii, jj, kk, ll, *lst, siz, cnt;
    long nat, **con, *nel;

    if (PyArg_ParseTuple(args, "O", &object)) {
        otmp = PyObject_GetAttrString(object, "angl");
        siz  = PyList_Size(otmp);
        lst  = (long*) malloc(3 * siz * sizeof(long));
        for (i = 0; i < siz; i++)
            for (j = 0; j < 3; j++)
                lst[3*i+j] = PyLong_AsLong(
                    PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        nat  = PyLong_AsLong(PyObject_GetAttrString(object, "natm"));
        otmp = PyObject_GetAttrString(object, "conn");
        con  = (long**) malloc(nat * sizeof(long*));
        nel  = (long*)  malloc(nat * sizeof(long));
        for (i = 0; i < nat; i++) {
            olst    = PyList_GetItem(otmp, i);
            nel[i]  = PyList_Size(olst);
            con[i]  = (long*) malloc(nel[i] * sizeof(long));
            for (j = 0; j < nel[i]; j++)
                con[i][j] = PyLong_AsLong(PyList_GetItem(olst, j));
        }
        Py_DECREF(otmp);

        out = PyList_New(0);
        for (i = 0; i < siz - 1; i++) {
            for (j = i + 1; j < siz; j++) {
                ii = -1; jj = -1; kk = -1; ll = -1;
                if      (lst[3*i+1]==lst[3*j]   && lst[3*i+2]==lst[3*j+1])
                    { ii=lst[3*i];   jj=lst[3*i+1]; kk=lst[3*i+2]; ll=lst[3*j+2]; }
                else if (lst[3*i+1]==lst[3*j+2] && lst[3*i+2]==lst[3*j+1])
                    { ii=lst[3*i];   jj=lst[3*i+1]; kk=lst[3*i+2]; ll=lst[3*j];   }
                else if (lst[3*i+1]==lst[3*j]   && lst[3*i]==lst[3*j+1])
                    { ii=lst[3*i+2]; jj=lst[3*i+1]; kk=lst[3*i];   ll=lst[3*j+2]; }
                else if (lst[3*i+1]==lst[3*j+2] && lst[3*i]==lst[3*j+1])
                    { ii=lst[3*i+2]; jj=lst[3*i+1]; kk=lst[3*i];   ll=lst[3*j];   }
                if (ii != -1 && jj != -1 && kk != -1 && ll != -1) {
                    cnt = 0;
                    for (k = 0; k < nel[ii]; k++) cnt += (ll == con[ii][k]);
                    if (cnt == 0) {
                        otmp = Py_BuildValue("[l,l,l,l]", ii, jj, kk, ll);
                        PyList_Append(out, otmp);
                        Py_DECREF(otmp);
                    }
                }
            }
        }
        free(lst); free(nel);
        for (i = 0; i < nat; i++) free(con[i]);
        free(con);
        return out;
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* =========================================================================
 * WRAPPERS PYTHON — con GPU (migrados de OpenCL a CUDA)
 * ====================================================================== */

/* -----------------------------------------------------------------
 * update_non_bonded — reconstruye la lista de pares no enlazados
 * ---------------------------------------------------------------- */
static PyObject* w_update_non_bonded(PyObject *self, PyObject *args) {
    PyObject        *out, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, cut, box[3], *itm;
    long            *bnd, *ang, *dih, n_bnd, n_ang, n_dih;
    long            i, j, k, *siz, *qms, *fre;

    if (PyArg_ParseTuple(args, "OO", &object, &molecule)) {
        init_cuda();

        otmp = PyObject_GetAttrString(object, "cut_list");
        cut  = PyFloat_AsDouble(otmp);
        if (cut > 0.0) cut *= cut; else cut = 1.0e99;
        Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(molecule, "boxl");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        for (i = 0; i < 3; i++) { itm = (double*) PyArray_GETPTR1(mtmp, i); box[i] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(molecule, "coor");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        siz  = PyArray_SHAPE(mtmp);
        xyz  = (double*) malloc(siz[0] * siz[1] * sizeof(double));
        for (k = 0, i = 0; i < siz[0]; i++)
            for (j = 0; j < siz[1]; j++) { itm = (double*) PyArray_GETPTR2(mtmp, i, j); xyz[k++] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "qmat");
        qms  = (long*) malloc(siz[0] * sizeof(long));
        for (i = 0; i < siz[0]; i++) qms[i] = PyObject_IsTrue(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "actv");
        fre  = (long*) malloc(siz[0] * sizeof(long));
        for (i = 0; i < siz[0]; i++) fre[i] = PyObject_IsTrue(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "bond");
        n_bnd = PyList_Size(otmp);
        bnd   = (long*) malloc(max(1, 2*n_bnd) * sizeof(long));
        for (i = 0; i < n_bnd; i++) {
            bnd[2*i]   = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 0));
            bnd[2*i+1] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 1));
        }
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "angl");
        n_ang = PyList_Size(otmp);
        ang   = (long*) malloc(max(1, 2*n_ang) * sizeof(long));
        for (i = 0; i < n_ang; i++) {
            ang[2*i]   = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 0));
            ang[2*i+1] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 2));
        }
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "dihe");
        n_dih = PyList_Size(otmp);
        dih   = (long*) malloc(max(1, 2*n_dih) * sizeof(long));
        for (i = 0; i < n_dih; i++) {
            dih[2*i]   = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 0));
            dih[2*i+1] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 3));
        }
        Py_DECREF(otmp);

        int max_pairs = siz[0] * 2048;

        /* --- Copiar datos al dispositivo ---------------------------------- */
        double* d_xyz  = (double*) cuda_malloc_copy(siz[0]*3*sizeof(double), xyz);
        long*   d_qms  = (long*)   cuda_malloc_copy(siz[0]*sizeof(long),     qms);
        long*   d_fre  = (long*)   cuda_malloc_copy(siz[0]*sizeof(long),     fre);
        long*   d_bnd  = (long*)   cuda_malloc_copy(max(1,2*n_bnd)*sizeof(long), bnd);
        long*   d_ang  = (long*)   cuda_malloc_copy(max(1,2*n_ang)*sizeof(long), ang);
        long*   d_dih  = (long*)   cuda_malloc_copy(max(1,2*n_dih)*sizeof(long), dih);
        long*   d_out  = (long*)   cuda_malloc(max_pairs * 2 * sizeof(long));
        int*    d_count = NULL;
        {
            int zero = 0;
            d_count = (int*) cuda_malloc_copy(sizeof(int), &zero);
        }

        /* --- Lanzar kernel 2D (igual padding que en OCL) ------------------- */
        dim3 block(16, 16);
        dim3 grid(((siz[0] + 15) / 16), ((siz[0] + 15) / 16));
        k_update_nb<<<grid, block>>>(
            d_xyz, d_qms, d_fre, d_bnd, d_ang, d_dih,
            d_out, d_count,
            (int)siz[0], box[0], box[1], box[2], cut,
            (int)n_bnd, (int)n_ang, (int)n_dih, max_pairs);

        /* --- Leer resultados ----------------------------------------------- */
        int count = 0;
        cuda_read(&count, d_count, sizeof(int));
        if (count > max_pairs) count = max_pairs;

        long* host_out = (long*) malloc(count * 2 * sizeof(long));
        cuda_read(host_out, d_out, count * 2 * sizeof(long));

        /* --- Construir lista Python ---------------------------------------- */
        out = PyList_New(count + n_dih);
        long idx = 0;
        for (i = 0; i < count; i++) {
            PyObject *sub = PyList_New(3);
            PyList_SET_ITEM(sub, 0, PyLong_FromLong(host_out[2*i]));
            PyList_SET_ITEM(sub, 1, PyLong_FromLong(host_out[2*i+1]));
            PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(1.0));
            PyList_SET_ITEM(out, idx++, sub);
        }
        for (i = 0; i < n_dih; i++) {
            PyObject *sub = PyList_New(3);
            PyList_SET_ITEM(sub, 0, PyLong_FromLong(dih[2*i]));
            PyList_SET_ITEM(sub, 1, PyLong_FromLong(dih[2*i+1]));
            PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(0.5));
            PyList_SET_ITEM(out, idx++, sub);
        }

        /* --- Liberar memoria ------------------------------------------------ */
        cudaFree(d_xyz); cudaFree(d_qms); cudaFree(d_fre);
        cudaFree(d_bnd); cudaFree(d_ang); cudaFree(d_dih);
        cudaFree(d_out); cudaFree(d_count);
        free(fre); free(qms); free(bnd); free(ang); free(dih); free(xyz); free(host_out);
        return out;
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* -----------------------------------------------------------------
 * energy_bond — energía total de enlaces
 * ---------------------------------------------------------------- */
static PyObject* w_energy_bond(PyObject *self, PyObject *args) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double           out_sum = 0.0;

    if (PyArg_ParseTuple(args, "OOO", &object, &molecule, &gradient)) {
        init_cuda();

        otmp = PyObject_GetAttrString(molecule, "coor");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        siz  = PyArray_SHAPE(mtmp);
        xyz  = (double*) malloc(siz[0] * siz[1] * sizeof(double));
        for (k = 0, i = 0; i < siz[0]; i++)
            for (j = 0; j < siz[1]; j++) { itm = (double*) PyArray_GETPTR2(mtmp, i, j); xyz[k++] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "actv");
        fre  = (long*) malloc(siz[0] * sizeof(long));
        for (i = 0; i < siz[0]; i++) fre[i] = PyObject_IsTrue(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString(object, "bond");
        n_lst = PyList_Size(otmp);
        if (n_lst == 0) { free(xyz); free(fre); return Py_BuildValue("d", 0.0); }

        lst = (long*) malloc(max(1, 2*n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++)
            for (j = 0; j < 2; j++)
                lst[2*i+j] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "bond_data");
        n_dat = PyList_Size(otmp);
        dat   = (double*) malloc(max(1, 2*n_dat) * sizeof(double));
        for (i = 0; i < n_dat; i++)
            for (j = 0; j < 2; j++)
                dat[2*i+j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "bond_indx");
        ind  = (long*) malloc(max(1, n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++) ind[i] = PyLong_AsLong(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        /* --- Copiar al dispositivo ----------------------------------------- */
        double* d_xyz     = (double*) cuda_malloc_copy(siz[0]*3*sizeof(double), xyz);
        long*   d_lst     = (long*)   cuda_malloc_copy(2*n_lst*sizeof(long),    lst);
        double* d_dat     = (double*) cuda_malloc_copy(2*n_dat*sizeof(double),  dat);
        long*   d_ind     = (long*)   cuda_malloc_copy(n_lst*sizeof(long),      ind);
        long*   d_fre     = (long*)   cuda_malloc_copy(siz[0]*sizeof(long),     fre);
        double* d_eng_out = (double*) cuda_malloc(n_lst * sizeof(double));
        double* d_grd_out = do_grad ? (double*) cuda_malloc(n_lst*6*sizeof(double)) : NULL;

        /* --- Lanzar kernel 1D --------------------------------------------- */
        int block = 64;
        int grid  = ((int)n_lst + 63) / 64;
        k_ebond<<<grid, block>>>(
            d_xyz, d_lst, d_dat, d_ind, d_fre,
            d_eng_out, do_grad ? d_grd_out : d_eng_out,
            do_grad, (int)n_lst);

        /* --- Leer resultados ----------------------------------------------- */
        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        cuda_read(host_eng, d_eng_out, n_lst * sizeof(double));

        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 6 * sizeof(double));
            cuda_read(host_grd, d_grd_out, n_lst * 6 * sizeof(double));
        }

        /* --- Acumular en Python -------------------------------------------- */
        otmp = PyObject_GetAttrString(molecule, "grad");
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);

        for (i = 0; i < n_lst; i++) {
            out_sum += host_eng[i];
            if (do_grad) {
                long ai = lst[2*i], aj = lst[2*i+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 0)) += host_grd[i*6+0];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 1)) += host_grd[i*6+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 2)) += host_grd[i*6+2];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 0)) += host_grd[i*6+3];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 1)) += host_grd[i*6+4];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 2)) += host_grd[i*6+5];
            }
        }
        if (do_grad) { Py_DECREF(mtmp_grad); Py_DECREF(otmp); free(host_grd); }

        /* --- Liberar memoria ------------------------------------------------ */
        free(host_eng);
        cudaFree(d_xyz); cudaFree(d_lst); cudaFree(d_dat);
        cudaFree(d_ind); cudaFree(d_fre); cudaFree(d_eng_out);
        if (d_grd_out) cudaFree(d_grd_out);
        free(xyz); free(lst); free(dat); free(ind); free(fre);
        return Py_BuildValue("d", out_sum);
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* -----------------------------------------------------------------
 * energy_angle — energía total de ángulos
 * ---------------------------------------------------------------- */
static PyObject* w_energy_angle(PyObject *self, PyObject *args) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double           out_sum = 0.0;

    if (PyArg_ParseTuple(args, "OOO", &object, &molecule, &gradient)) {
        init_cuda();

        otmp = PyObject_GetAttrString(molecule, "coor");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        siz  = PyArray_SHAPE(mtmp);
        xyz  = (double*) malloc(siz[0] * siz[1] * sizeof(double));
        for (k = 0, i = 0; i < siz[0]; i++)
            for (j = 0; j < siz[1]; j++) { itm = (double*) PyArray_GETPTR2(mtmp, i, j); xyz[k++] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "actv");
        fre  = (long*) malloc(siz[0] * sizeof(long));
        for (i = 0; i < siz[0]; i++) fre[i] = PyObject_IsTrue(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString(object, "angl");
        n_lst = PyList_Size(otmp);
        if (n_lst == 0) { free(xyz); free(fre); return Py_BuildValue("d", 0.0); }

        lst = (long*) malloc(max(1, 3*n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++)
            for (j = 0; j < 3; j++)
                lst[3*i+j] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "angl_data");
        n_dat = PyList_Size(otmp);
        dat   = (double*) malloc(max(1, 2*n_dat) * sizeof(double));
        for (i = 0; i < n_dat; i++)
            for (j = 0; j < 2; j++)
                dat[2*i+j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "angl_indx");
        ind  = (long*) malloc(max(1, n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++) ind[i] = PyLong_AsLong(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        /* --- Dispositivo --------------------------------------------------- */
        double* d_xyz     = (double*) cuda_malloc_copy(siz[0]*3*sizeof(double), xyz);
        long*   d_lst     = (long*)   cuda_malloc_copy(3*n_lst*sizeof(long),    lst);
        double* d_dat     = (double*) cuda_malloc_copy(2*n_dat*sizeof(double),  dat);
        long*   d_ind     = (long*)   cuda_malloc_copy(n_lst*sizeof(long),      ind);
        long*   d_fre     = (long*)   cuda_malloc_copy(siz[0]*sizeof(long),     fre);
        double* d_eng_out = (double*) cuda_malloc(n_lst * sizeof(double));
        double* d_grd_out = do_grad ? (double*) cuda_malloc(n_lst*9*sizeof(double)) : NULL;

        int block = 64, grid = ((int)n_lst + 63) / 64;
        k_eangle<<<grid, block>>>(
            d_xyz, d_lst, d_dat, d_ind, d_fre,
            d_eng_out, do_grad ? d_grd_out : d_eng_out,
            do_grad, (int)n_lst);

        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        cuda_read(host_eng, d_eng_out, n_lst * sizeof(double));

        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 9 * sizeof(double));
            cuda_read(host_grd, d_grd_out, n_lst * 9 * sizeof(double));
        }

        otmp = PyObject_GetAttrString(molecule, "grad");
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);

        for (i = 0; i < n_lst; i++) {
            out_sum += host_eng[i];
            if (do_grad) {
                long ai = lst[3*i], aj = lst[3*i+1], ak = lst[3*i+2];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 0)) += host_grd[i*9+0];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 1)) += host_grd[i*9+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 2)) += host_grd[i*9+2];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 0)) += host_grd[i*9+3];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 1)) += host_grd[i*9+4];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 2)) += host_grd[i*9+5];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 0)) += host_grd[i*9+6];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 1)) += host_grd[i*9+7];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 2)) += host_grd[i*9+8];
            }
        }
        if (do_grad) { Py_DECREF(mtmp_grad); Py_DECREF(otmp); free(host_grd); }

        free(host_eng);
        cudaFree(d_xyz); cudaFree(d_lst); cudaFree(d_dat);
        cudaFree(d_ind); cudaFree(d_fre); cudaFree(d_eng_out);
        if (d_grd_out) cudaFree(d_grd_out);
        free(xyz); free(lst); free(dat); free(ind); free(fre);
        return Py_BuildValue("d", out_sum);
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* -----------------------------------------------------------------
 * energy_dihedral — energía total de diedros
 * ---------------------------------------------------------------- */
static PyObject* w_energy_dihedral(PyObject *self, PyObject *args) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double           out_sum = 0.0;

    if (PyArg_ParseTuple(args, "OOO", &object, &molecule, &gradient)) {
        init_cuda();

        otmp = PyObject_GetAttrString(molecule, "coor");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        siz  = PyArray_SHAPE(mtmp);
        xyz  = (double*) malloc(siz[0] * siz[1] * sizeof(double));
        for (k = 0, i = 0; i < siz[0]; i++)
            for (j = 0; j < siz[1]; j++) { itm = (double*) PyArray_GETPTR2(mtmp, i, j); xyz[k++] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "actv");
        fre  = (long*) malloc(siz[0] * sizeof(long));
        for (i = 0; i < siz[0]; i++) fre[i] = PyObject_IsTrue(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString(object, "dihe");
        n_lst = PyList_Size(otmp);
        if (n_lst == 0) { free(xyz); free(fre); return Py_BuildValue("d", 0.0); }

        lst = (long*) malloc(max(1, 4*n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++)
            for (j = 0; j < 4; j++)
                lst[4*i+j] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "dihe_data");
        n_dat = PyList_Size(otmp);
        dat   = (double*) malloc(max(1, 12*n_dat) * sizeof(double));
        for (i = 0; i < 12*n_dat; i++) dat[i] = 0.0;
        for (i = 0; i < n_dat; i++)
            for (j = 0; j < 12; j++)
                dat[12*i+j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(otmp, i), j));
        Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "dihe_indx");
        ind  = (long*) malloc(max(1, n_lst) * sizeof(long));
        for (i = 0; i < n_lst; i++) ind[i] = PyLong_AsLong(PyList_GetItem(otmp, i));
        Py_DECREF(otmp);

        /* --- Dispositivo --------------------------------------------------- */
        double* d_xyz     = (double*) cuda_malloc_copy(siz[0]*3*sizeof(double),  xyz);
        long*   d_lst     = (long*)   cuda_malloc_copy(4*n_lst*sizeof(long),     lst);
        double* d_dat     = (double*) cuda_malloc_copy(12*n_dat*sizeof(double),  dat);
        long*   d_ind     = (long*)   cuda_malloc_copy(n_lst*sizeof(long),       ind);
        long*   d_fre     = (long*)   cuda_malloc_copy(siz[0]*sizeof(long),      fre);
        double* d_eng_out = (double*) cuda_malloc(n_lst * sizeof(double));
        double* d_grd_out = do_grad ? (double*) cuda_malloc(n_lst*12*sizeof(double)) : NULL;

        int block = 64, grid = ((int)n_lst + 63) / 64;
        k_edihedral<<<grid, block>>>(
            d_xyz, d_lst, d_dat, d_ind, d_fre,
            d_eng_out, do_grad ? d_grd_out : d_eng_out,
            do_grad, (int)n_lst);

        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        cuda_read(host_eng, d_eng_out, n_lst * sizeof(double));

        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 12 * sizeof(double));
            cuda_read(host_grd, d_grd_out, n_lst * 12 * sizeof(double));
        }

        otmp = PyObject_GetAttrString(molecule, "grad");
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);

        for (i = 0; i < n_lst; i++) {
            out_sum += host_eng[i];
            if (do_grad) {
                long ai = lst[4*i], aj = lst[4*i+1],
                     ak = lst[4*i+2], al = lst[4*i+3];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 0)) += host_grd[i*12+0];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 1)) += host_grd[i*12+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 2)) += host_grd[i*12+2];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 0)) += host_grd[i*12+3];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 1)) += host_grd[i*12+4];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 2)) += host_grd[i*12+5];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 0)) += host_grd[i*12+6];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 1)) += host_grd[i*12+7];
                *((double*) PyArray_GETPTR2(mtmp_grad, ak, 2)) += host_grd[i*12+8];
                *((double*) PyArray_GETPTR2(mtmp_grad, al, 0)) += host_grd[i*12+9];
                *((double*) PyArray_GETPTR2(mtmp_grad, al, 1)) += host_grd[i*12+10];
                *((double*) PyArray_GETPTR2(mtmp_grad, al, 2)) += host_grd[i*12+11];
            }
        }
        if (do_grad) { Py_DECREF(mtmp_grad); Py_DECREF(otmp); free(host_grd); }

        free(host_eng);
        cudaFree(d_xyz); cudaFree(d_lst); cudaFree(d_dat);
        cudaFree(d_ind); cudaFree(d_fre); cudaFree(d_eng_out);
        if (d_grd_out) cudaFree(d_grd_out);
        free(xyz); free(lst); free(dat); free(ind); free(fre);
        return Py_BuildValue("d", out_sum);
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* -----------------------------------------------------------------
 * energy_non_bonded — energía electrostática + Lennard-Jones
 * ---------------------------------------------------------------- */
static PyObject* w_energy_non_bonded(PyObject *self, PyObject *args) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *dat, *scl, *itm, *qms;
    long            *siz, i, j, k;
    long            *lst, n_lst;
    double           oel_sum = 0.0, olj_sum = 0.0, con, cof, box[3], epsi, epsf;

    if (PyArg_ParseTuple(args, "OOOd", &object, &molecule, &gradient, &epsi)) {
        init_cuda();

        otmp = PyObject_GetAttrString(molecule, "boxl");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        for (i = 0; i < 3; i++) { itm = (double*) PyArray_GETPTR1(mtmp, i); box[i] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(molecule, "coor");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        siz  = PyArray_SHAPE(mtmp);
        xyz  = (double*) malloc(siz[0] * siz[1] * sizeof(double));
        for (k = 0, i = 0; i < siz[0]; i++)
            for (j = 0; j < siz[1]; j++) { itm = (double*) PyArray_GETPTR2(mtmp, i, j); xyz[k++] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(object, "qmat");
        qms  = (double*) malloc(siz[0] * sizeof(double));
        for (i = 0; i < siz[0]; i++) qms[i] = PyObject_IsTrue(PyList_GetItem(otmp, i)) ? 0.0 : 1.0;
        Py_DECREF(otmp);

        dat  = (double*) malloc(3 * siz[0] * sizeof(double));

        otmp = PyObject_GetAttrString(molecule, "epsi");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        for (i = 0; i < siz[0]; i++) { itm = (double*) PyArray_GETPTR1(mtmp, i); dat[3*i] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(molecule, "rmin");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        for (i = 0; i < siz[0]; i++) { itm = (double*) PyArray_GETPTR1(mtmp, i); dat[3*i+1] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        otmp = PyObject_GetAttrString(molecule, "chrg");
        mtmp = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);
        for (i = 0; i < siz[0]; i++) { itm = (double*) PyArray_GETPTR1(mtmp, i); dat[3*i+2] = *itm; }
        Py_DECREF(mtmp); Py_DECREF(otmp);

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp = PyObject_GetAttrString(object, "cut_on");  con = PyFloat_AsDouble(otmp); Py_DECREF(otmp);
        otmp = PyObject_GetAttrString(object, "cut_off"); cof = PyFloat_AsDouble(otmp); Py_DECREF(otmp);

        otmp  = PyObject_GetAttrString(object, "nbnd");
        n_lst = PyList_Size(otmp);
        if (n_lst == 0) {
            free(xyz); free(dat); free(qms);
            return Py_BuildValue("(d,d)", 0.0, 0.0);
        }

        lst = (long*)   malloc(max(1, 2*n_lst) * sizeof(long));
        scl = (double*) malloc(max(1,   n_lst) * sizeof(double));
        for (i = 0; i < n_lst; i++) {
            lst[2*i]   = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 0));
            lst[2*i+1] = PyLong_AsLong(PyList_GetItem(PyList_GetItem(otmp, i), 1));
            scl[i]     = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(otmp, i), 2));
        }
        Py_DECREF(otmp);

        epsf = 1389.35484620709144110151 / epsi;

        /* --- Copiar al dispositivo ----------------------------------------- */
        double* d_xyz     = (double*) cuda_malloc_copy(siz[0]*3*sizeof(double), xyz);
        long*   d_lst     = (long*)   cuda_malloc_copy(2*n_lst*sizeof(long),    lst);
        double* d_dat     = (double*) cuda_malloc_copy(3*siz[0]*sizeof(double), dat);
        double* d_qms     = (double*) cuda_malloc_copy(siz[0]*sizeof(double),   qms);
        double* d_scl     = (double*) cuda_malloc_copy(n_lst*sizeof(double),    scl);
        double* d_oel_out = (double*) cuda_malloc(n_lst * sizeof(double));
        double* d_olj_out = (double*) cuda_malloc(n_lst * sizeof(double));
        double* d_grd_out = do_grad ? (double*) cuda_malloc(n_lst*6*sizeof(double)) : NULL;

        int block = 64, grid = ((int)n_lst + 63) / 64;

        if (con > 0.0 && cof > con) {
            /* Con cutoff suavizado */
            double c2on = con*con, c2of = cof*cof;
            double _g   = pow(c2of - c2on, 3.0);
            double _a   = c2of*c2of*(c2of - 3.0*c2on) / _g;
            double _b   = 6.0*c2of*c2on / _g;
            double _c   = -(c2of + c2on) / _g;
            double _d   = 0.4 / _g;
            double _el1 = 8.0*(c2of*c2on*(cof - con) -
                               0.2*(cof*c2of*c2of - con*c2on*c2on)) / _g;
            double _el2 = -_a/cof + _b*cof + _c*cof*c2of + _d*cof*c2of*c2of;
            double k6   = (cof*c2of) / (cof*c2of - con*c2on);
            double k12  = pow(c2of, 3.0) / (pow(c2of, 3.0) - pow(c2on, 3.0));

            k_enonbonded_cut<<<grid, block>>>(
                d_xyz, d_lst, d_dat, d_qms, d_scl,
                d_oel_out, d_olj_out, do_grad ? d_grd_out : d_oel_out,
                box[0], box[1], box[2], epsf,
                c2on, c2of, con, cof,
                _a, _b, _c, _d, _el1, _el2, k6, k12,
                do_grad, (int)n_lst);
        } else {
            /* Sin cutoff */
            k_enonbonded_nocut<<<grid, block>>>(
                d_xyz, d_lst, d_dat, d_qms, d_scl,
                d_oel_out, d_olj_out, do_grad ? d_grd_out : d_oel_out,
                box[0], box[1], box[2], epsf,
                do_grad, (int)n_lst);
        }

        /* --- Leer resultados ----------------------------------------------- */
        double* host_oel = (double*) malloc(n_lst * sizeof(double));
        double* host_olj = (double*) malloc(n_lst * sizeof(double));
        cuda_read(host_oel, d_oel_out, n_lst * sizeof(double));
        cuda_read(host_olj, d_olj_out, n_lst * sizeof(double));

        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 6 * sizeof(double));
            cuda_read(host_grd, d_grd_out, n_lst * 6 * sizeof(double));
        }

        otmp = PyObject_GetAttrString(molecule, "grad");
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT(otmp, NPY_DOUBLE);

        for (i = 0; i < n_lst; i++) {
            oel_sum += host_oel[i];
            olj_sum += host_olj[i];
            if (do_grad) {
                long ai = lst[2*i], aj = lst[2*i+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 0)) += host_grd[i*6+0];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 1)) += host_grd[i*6+1];
                *((double*) PyArray_GETPTR2(mtmp_grad, ai, 2)) += host_grd[i*6+2];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 0)) += host_grd[i*6+3];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 1)) += host_grd[i*6+4];
                *((double*) PyArray_GETPTR2(mtmp_grad, aj, 2)) += host_grd[i*6+5];
            }
        }
        if (do_grad) { Py_DECREF(mtmp_grad); Py_DECREF(otmp); free(host_grd); }

        free(host_oel); free(host_olj);
        cudaFree(d_xyz); cudaFree(d_lst); cudaFree(d_dat);
        cudaFree(d_qms); cudaFree(d_scl);
        cudaFree(d_oel_out); cudaFree(d_olj_out);
        if (d_grd_out) cudaFree(d_grd_out);
        free(xyz); free(dat); free(lst); free(scl); free(qms);
        return Py_BuildValue("(d,d)", oel_sum, olj_sum);
    } else { Py_INCREF(Py_None); return Py_None; }
}

/* =========================================================================
 * REGISTRO DEL MÓDULO PYTHON
 * ====================================================================== */

static struct PyMethodDef methods[] = {
    { "guess_angles",      (PyCFunction)w_guess_angles,       METH_VARARGS },
    { "guess_dihedrals",   (PyCFunction)w_guess_dihedrals,    METH_VARARGS },
    { "update_non_bonded", (PyCFunction)w_update_non_bonded,  METH_VARARGS },
    { "ebond",             (PyCFunction)w_energy_bond,        METH_VARARGS },
    { "eangle",            (PyCFunction)w_energy_angle,       METH_VARARGS },
    { "edihedral",         (PyCFunction)w_energy_dihedral,    METH_VARARGS },
    { "enonbonded",        (PyCFunction)w_energy_non_bonded,  METH_VARARGS },
    { 0, 0, 0 }
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "_molmech",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__molmech(void) {
    PyObject *my_module;
    my_module = PyModule_Create(&moddef);
    import_array();
    return my_module;
}
