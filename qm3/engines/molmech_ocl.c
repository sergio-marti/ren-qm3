#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static cl_kernel k_ebond = NULL;
static cl_kernel k_eangle = NULL;
static cl_kernel k_edihedral = NULL;
static cl_kernel k_enonbonded_cut = NULL;
static cl_kernel k_enonbonded_nocut = NULL;
static cl_kernel k_update_nb = NULL;

const char* ocl_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void ebond(__global const double* xyz, __global const long* lst, __global const double* dat, "
"                    __global const long* ind, __global const long* fre, "
"                    __global double* out_eng, __global double* out_grd, int do_grad, int n_lst) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n_lst) return;\n"
"    long ai = lst[2*i] * 3, aj = lst[2*i+1] * 3;\n"
"    if (!fre[lst[2*i]] && !fre[lst[2*i+1]]) {\n"
"        out_eng[i] = 0.0;\n"
"        if (do_grad) { for(int j=0; j<6; j++) out_grd[i*6+j] = 0.0; }\n"
"        return;\n"
"    }\n"
"    double vec[3], val, dif, tmp;\n"
"    vec[0] = xyz[ai] - xyz[aj]; vec[1] = xyz[ai+1] - xyz[aj+1]; vec[2] = xyz[ai+2] - xyz[aj+2];\n"
"    val = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);\n"
"    dif = val - dat[ind[i]*2+1];\n"
"    tmp = dif * dat[ind[i]*2];\n"
"    out_eng[i] = 0.5 * tmp * dif;\n"
"    if (do_grad) {\n"
"        tmp *= 1.0 / val;\n"
"        out_grd[i*6+0] = tmp * vec[0]; out_grd[i*6+1] = tmp * vec[1]; out_grd[i*6+2] = tmp * vec[2];\n"
"        out_grd[i*6+3] = -tmp * vec[0]; out_grd[i*6+4] = -tmp * vec[1]; out_grd[i*6+5] = -tmp * vec[2];\n"
"    }\n"
"}\n"

"__kernel void eangle(__global const double* xyz, __global const long* lst, __global const double* dat, "
"                     __global const long* ind, __global const long* fre, "
"                     __global double* out_eng, __global double* out_grd, int do_grad, int n_lst) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n_lst) return;\n"
"    long idx_i = lst[3*i], idx_j = lst[3*i+1], idx_k = lst[3*i+2];\n"
"    long ai = idx_i*3, aj = idx_j*3, ak = idx_k*3;\n"
"    if (!fre[idx_i] && !fre[idx_j] && !fre[idx_k]) {\n"
"        out_eng[i] = 0.0;\n"
"        if(do_grad) { for(int j=0; j<9; j++) out_grd[i*9+j] = 0.0; }\n"
"        return;\n"
"    }\n"
"    double dij[3], dkj[3], dti[3], dtj[3], dtk[3];\n"
"    dij[0] = xyz[ai] - xyz[aj]; dij[1] = xyz[ai+1] - xyz[aj+1]; dij[2] = xyz[ai+2] - xyz[aj+2];\n"
"    double rij = sqrt(dij[0]*dij[0] + dij[1]*dij[1] + dij[2]*dij[2]);\n"
"    dij[0]/=rij; dij[1]/=rij; dij[2]/=rij;\n"
"    dkj[0] = xyz[ak] - xyz[aj]; dkj[1] = xyz[ak+1] - xyz[aj+1]; dkj[2] = xyz[ak+2] - xyz[aj+2];\n"
"    double rkj = sqrt(dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2]);\n"
"    dkj[0]/=rkj; dkj[1]/=rkj; dkj[2]/=rkj;\n"
"    double fac = dij[0]*dkj[0] + dij[1]*dkj[1] + dij[2]*dkj[2];\n"
"    double abs_fac = fabs(fac);\n"
"    fac = fmin(abs_fac, 1.0 - 1.0e-6) * (fac < 0 ? -1.0 : 1.0);\n"
"    double val = acos(fac);\n"
"    double dif = val - dat[ind[i]*2+1];\n"
"    double tmp = dif * dat[ind[i]*2];\n"
"    out_eng[i] = 0.5 * tmp * dif;\n"
"    if (do_grad) {\n"
"        tmp *= -1.0 / sqrt(1.0 - fac * fac);\n"
"        for(int j=0; j<3; j++) {\n"
"            dti[j] = (dkj[j] - fac * dij[j]) / rij;\n"
"            dtk[j] = (dij[j] - fac * dkj[j]) / rkj;\n"
"            dtj[j] = -(dti[j] + dtk[j]);\n"
"        }\n"
"        out_grd[i*9+0] = tmp*dti[0]; out_grd[i*9+1] = tmp*dti[1]; out_grd[i*9+2] = tmp*dti[2];\n"
"        out_grd[i*9+3] = tmp*dtj[0]; out_grd[i*9+4] = tmp*dtj[1]; out_grd[i*9+5] = tmp*dtj[2];\n"
"        out_grd[i*9+6] = tmp*dtk[0]; out_grd[i*9+7] = tmp*dtk[1]; out_grd[i*9+8] = tmp*dtk[2];\n"
"    }\n"
"}\n"

"__kernel void edihedral(__global const double* xyz, __global const long* lst, __global const double* dat, "
"                        __global const long* ind, __global const long* fre, "
"                        __global double* out_eng, __global double* out_grd, int do_grad, int n_lst) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n_lst) return;\n"
"    long idx_i=lst[4*i], idx_j=lst[4*i+1], idx_k=lst[4*i+2], idx_l=lst[4*i+3];\n"
"    long ai=idx_i*3, aj=idx_j*3, ak=idx_k*3, al=idx_l*3;\n"
"    if (!fre[idx_i] && !fre[idx_j] && !fre[idx_k] && !fre[idx_l]) {\n"
"        out_eng[i] = 0.0;\n"
"        if(do_grad) { for(int j=0; j<12; j++) out_grd[i*12+j] = 0.0; }\n"
"        return;\n"
"    }\n"
"    double dji[3], dkj[3], dlk[3], vt[3], vu[3], vtu[3], dki[3], dlj[3], dvt[3], dvu[3];\n"
"    for(int j=0; j<3; j++) { dji[j]=xyz[aj+j]-xyz[ai+j]; dkj[j]=xyz[ak+j]-xyz[aj+j]; dlk[j]=xyz[al+j]-xyz[ak+j]; }\n"
"    double rkj = sqrt(dkj[0]*dkj[0] + dkj[1]*dkj[1] + dkj[2]*dkj[2]);\n"
"    vt[0]=dji[1]*dkj[2]-dkj[1]*dji[2]; vt[1]=dji[2]*dkj[0]-dkj[2]*dji[0]; vt[2]=dji[0]*dkj[1]-dkj[0]*dji[1];\n"
"    double rt2 = vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2];\n"
"    vu[0]=dkj[1]*dlk[2]-dlk[1]*dkj[2]; vu[1]=dkj[2]*dlk[0]-dlk[2]*dkj[0]; vu[2]=dkj[0]*dlk[1]-dlk[0]*dkj[1];\n"
"    double ru2 = vu[0]*vu[0] + vu[1]*vu[1] + vu[2]*vu[2];\n"
"    vtu[0]=vt[1]*vu[2]-vu[1]*vt[2]; vtu[1]=vt[2]*vu[0]-vu[2]*vt[0]; vtu[2]=vt[0]*vu[1]-vu[0]*vt[1];\n"
"    double rtu = sqrt(rt2 * ru2);\n"
"    if(rtu == 0.0) { out_eng[i]=0.0; if(do_grad) { for(int j=0;j<12;j++) out_grd[i*12+j]=0.0; } return; }\n"
"    double cs1=0.0, sn1=0.0;\n"
"    for(int j=0; j<3; j++) { cs1 += vt[j]*vu[j]; sn1 += dkj[j]*vtu[j]; }\n"
"    cs1 /= rtu; sn1 /= (rtu * rkj);\n"
"    double cs2=cs1*cs1-sn1*sn1, sn2=2.0*cs1*sn1;\n"
"    double cs3=cs1*cs2-sn1*sn2, sn3=cs1*sn2+sn1*cs2;\n"
"    double cs4=cs1*cs3-sn1*sn3, sn4=cs1*sn3+sn1*cs3;\n"
"    double cs5=cs1*cs4-sn1*sn4, sn5=cs1*sn4+sn1*cs4;\n"
"    double cs6=cs1*cs5-sn1*sn5, sn6=cs1*sn5+sn1*cs5;\n"
"    double dph = 0.0, eng = 0.0, cd, sd;\n"
"    int idx = ind[i]*12;\n"
"    double c_cs[6] = {cs1, cs2, cs3, cs4, cs5, cs6};\n"
"    double c_sn[6] = {sn1, sn2, sn3, sn4, sn5, sn6};\n"
"    for(int k=0; k<6; k++) {\n"
"        if(dat[idx+k*2] != 0.0) {\n"
"            cd = cos(dat[idx+k*2+1]); sd = sin(dat[idx+k*2+1]);\n"
"            dph += dat[idx+k*2] * (k+1.0) * (c_cs[k]*sd - c_sn[k]*cd);\n"
"            eng += dat[idx+k*2] * (1.0 + c_cs[k]*cd + c_sn[k]*sd);\n"
"        }\n"
"    }\n"
"    out_eng[i] = eng;\n"
"    if(do_grad) {\n"
"        for(int j=0; j<3; j++) { dki[j]=xyz[ak+j]-xyz[ai+j]; dlj[j]=xyz[al+j]-xyz[aj+j]; }\n"
"        dvt[0]=(vt[1]*dkj[2]-dkj[1]*vt[2])/(rt2*rkj); dvt[1]=(vt[2]*dkj[0]-dkj[2]*vt[0])/(rt2*rkj); dvt[2]=(vt[0]*dkj[1]-dkj[0]*vt[1])/(rt2*rkj);\n"
"        dvu[0]=(vu[1]*dkj[2]-dkj[1]*vu[2])/(ru2*rkj); dvu[1]=(vu[2]*dkj[0]-dkj[2]*vu[0])/(ru2*rkj); dvu[2]=(vu[0]*dkj[1]-dkj[0]*vu[1])/(ru2*rkj);\n"
"        out_grd[i*12+0] = (dkj[2]*dvt[1] - dkj[1]*dvt[2]) * dph;\n"
"        out_grd[i*12+1] = (dkj[0]*dvt[2] - dkj[2]*dvt[0]) * dph;\n"
"        out_grd[i*12+2] = (dkj[1]*dvt[0] - dkj[0]*dvt[1]) * dph;\n"
"        out_grd[i*12+3] = (dki[1]*dvt[2] - dki[2]*dvt[1] - dlk[2]*dvu[1] + dlk[1]*dvu[2]) * dph;\n"
"        out_grd[i*12+4] = (dki[2]*dvt[0] - dki[0]*dvt[2] - dlk[0]*dvu[2] + dlk[2]*dvu[0]) * dph;\n"
"        out_grd[i*12+5] = (dki[0]*dvt[1] - dki[1]*dvt[0] - dlk[1]*dvu[0] + dlk[0]*dvu[1]) * dph;\n"
"        out_grd[i*12+6] = (dji[2]*dvt[1] - dji[1]*dvt[2] - dlj[1]*dvu[2] + dlj[2]*dvu[1]) * dph;\n"
"        out_grd[i*12+7] = (dji[0]*dvt[2] - dji[2]*dvt[0] - dlj[2]*dvu[0] + dlj[0]*dvu[2]) * dph;\n"
"        out_grd[i*12+8] = (dji[1]*dvt[0] - dji[0]*dvt[1] - dlj[0]*dvu[1] + dlj[1]*dvu[0]) * dph;\n"
"        out_grd[i*12+9] = (-dkj[2]*dvu[1] + dkj[1]*dvu[2]) * dph;\n"
"        out_grd[i*12+10]= (-dkj[0]*dvu[2] + dkj[2]*dvu[0]) * dph;\n"
"        out_grd[i*12+11]= (-dkj[1]*dvu[0] + dkj[0]*dvu[1]) * dph;\n"
"    }\n"
"}\n"

"__kernel void enonbonded_cut(__global const double* xyz, __global const long* lst, __global const double* dat, "
"                             __global const double* qms, __global const double* scl, "
"                             __global double* out_oel, __global double* out_olj, __global double* out_grd, "
"                             double box0, double box1, double box2, double epsf, "
"                             double c2on, double c2of, double con, double cof, "
"                             double _a, double _b, double _c, double _d, double _el1, double _el2, "
"                             double k6, double k12, int do_grad, int n_lst) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n_lst) return;\n"
"    long ii = lst[2*i], jj = lst[2*i+1];\n"
"    long ai = 3*ii, aj = 3*jj;\n"
"    double dr[3];\n"
"    dr[0] = xyz[ai] - xyz[aj]; dr[1] = xyz[ai+1] - xyz[aj+1]; dr[2] = xyz[ai+2] - xyz[aj+2];\n"
"    if (box0 > 0.0) dr[0] -= box0 * round(dr[0]/box0);\n"
"    if (box1 > 0.0) dr[1] -= box1 * round(dr[1]/box1);\n"
"    if (box2 > 0.0) dr[2] -= box2 * round(dr[2]/box2);\n"
"    double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];\n"
"    if (r2 > c2of) { out_oel[i]=0.0; out_olj[i]=0.0; if(do_grad) { for(int j=0;j<6;j++) out_grd[i*6+j]=0.0; } return; }\n"
"    double eij = dat[3*ii] * dat[3*jj];\n"
"    double sij = dat[3*ii+1] + dat[3*jj+1];\n"
"    double qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];\n"
"    double r = sqrt(r2), s = 1.0/r;\n"
"    double s3 = (sij*s)*(sij*s)*(sij*s), s6 = s3*s3, s12 = s6*s6;\n"
"    double _oel=0.0, _olj=0.0, df=0.0, tmp=0.0;\n"
"    if (r2 <= c2on) {\n"
"        tmp = qij * s;\n"
"        _oel = scl[i] * (tmp + qij * _el1);\n"
"        double _lj1 = (sij/cof)*(sij/con)*(sij/cof)*(sij/con)*(sij/cof)*(sij/con);\n"
"        double _lj2 = _lj1 * _lj1;\n"
"        _olj = scl[i] * eij * ((s12 - _lj2) - 2.0 * (s6 - _lj1));\n"
"        df = (12.0 * eij * (s6 - s12) - tmp) / r2;\n"
"    } else {\n"
"        double r3 = r*r2, r5 = r3*r2;\n"
"        _oel = scl[i] * qij * (_a*s - _b*r - _c*r3 - _d*r5 + _el2);\n"
"        double _lj1 = (sij/cof)*(sij/cof)*(sij/cof);\n"
"        double _lj2 = _lj1 * _lj1;\n"
"        _olj = scl[i] * eij * (k12 * (s6-_lj2)*(s6-_lj2) - 2.0 * k6 * (s3-_lj1)*(s3-_lj1));\n"
"        df = -qij * (_a/r3 + _b*s + 3.0*_c*r + 5.0*_d*r3);\n"
"        df -= 12.0 * eij * (k12 * s6 * (s6 - _lj2) - k6 * s3 * (s3 - _lj1)) / r2;\n"
"    }\n"
"    out_oel[i] = _oel; out_olj[i] = _olj;\n"
"    if (do_grad) {\n"
"        out_grd[i*6+0] = scl[i]*df*dr[0]; out_grd[i*6+1] = scl[i]*df*dr[1]; out_grd[i*6+2] = scl[i]*df*dr[2];\n"
"        out_grd[i*6+3] = -scl[i]*df*dr[0]; out_grd[i*6+4] = -scl[i]*df*dr[1]; out_grd[i*6+5] = -scl[i]*df*dr[2];\n"
"    }\n"
"}\n"

"__kernel void enonbonded_nocut(__global const double* xyz, __global const long* lst, __global const double* dat, "
"                               __global const double* qms, __global const double* scl, "
"                               __global double* out_oel, __global double* out_olj, __global double* out_grd, "
"                               double box0, double box1, double box2, double epsf, "
"                               int do_grad, int n_lst) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= n_lst) return;\n"
"    long ii = lst[2*i], jj = lst[2*i+1];\n"
"    long ai = 3*ii, aj = 3*jj;\n"
"    double dr[3];\n"
"    dr[0] = xyz[ai] - xyz[aj]; dr[1] = xyz[ai+1] - xyz[aj+1]; dr[2] = xyz[ai+2] - xyz[aj+2];\n"
"    if (box0 > 0.0) dr[0] -= box0 * round(dr[0]/box0);\n"
"    if (box1 > 0.0) dr[1] -= box1 * round(dr[1]/box1);\n"
"    if (box2 > 0.0) dr[2] -= box2 * round(dr[2]/box2);\n"
"    double r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];\n"
"    double eij = dat[3*ii] * dat[3*jj];\n"
"    double sij = dat[3*ii+1] + dat[3*jj+1];\n"
"    double qij = dat[3*ii+2] * dat[3*jj+2] * epsf * qms[ii] * qms[jj];\n"
"    double s = 1.0 / sqrt(r2);\n"
"    double s6 = (sij*s)*(sij*s)*(sij*s)*(sij*s)*(sij*s)*(sij*s);\n"
"    double tmp = qij * s;\n"
"    out_oel[i] = scl[i] * tmp;\n"
"    out_olj[i] = scl[i] * eij * s6 * (s6 - 2.0);\n"
"    if (do_grad) {\n"
"        double df = scl[i] * (12.0 * eij * s6 * (1.0 - s6) - tmp) / r2;\n"
"        out_grd[i*6+0] = df*dr[0]; out_grd[i*6+1] = df*dr[1]; out_grd[i*6+2] = df*dr[2];\n"
"        out_grd[i*6+3] = -df*dr[0]; out_grd[i*6+4] = -df*dr[1]; out_grd[i*6+5] = -df*dr[2];\n"
"    }\n"
"}\n"

"__kernel void update_nb(__global const double* xyz, __global const long* qms, __global const long* fre, "
"                        __global const long* bnd, __global const long* ang, __global const long* dih, "
"                        __global long* out_pairs, volatile __global int* pair_count, "
"                        int siz, double box0, double box1, double box2, double cut, "
"                        int n_bnd, int n_ang, int n_dih, int max_pairs) {\n"
"    int i = get_global_id(0);\n"
"    int j = get_global_id(1);\n"
"    if (i >= siz || j >= siz || i >= j) return;\n"
"    if ((qms[i] == 1 && qms[j] == 1) || (fre[i] == 0 && fre[j] == 0)) return;\n"
"    double dr0 = xyz[i*3] - xyz[j*3]; if (box0 > 0.0) dr0 -= box0 * round(dr0/box0);\n"
"    double dr1 = xyz[i*3+1] - xyz[j*3+1]; if (box1 > 0.0) dr1 -= box1 * round(dr1/box1);\n"
"    double dr2 = xyz[i*3+2] - xyz[j*3+2]; if (box2 > 0.0) dr2 -= box2 * round(dr2/box2);\n"
"    if (dr0*dr0 + dr1*dr1 + dr2*dr2 <= cut) {\n"
"        int f = 0;\n"
"        for(int k=0; k<n_bnd && !f; k++) { f |= ((i==bnd[2*k] && j==bnd[2*k+1]) || (i==bnd[2*k+1] && j==bnd[2*k])); }\n"
"        for(int k=0; k<n_ang && !f; k++) { f |= ((i==ang[2*k] && j==ang[2*k+1]) || (i==ang[2*k+1] && j==ang[2*k])); }\n"
"        for(int k=0; k<n_dih && !f; k++) { f |= ((i==dih[2*k] && j==dih[2*k+1]) || (i==dih[2*k+1] && j==dih[2*k])); }\n"
"        if (!f) {\n"
"            int idx = atomic_inc(pair_count);\n"
"            if (idx < max_pairs) {\n"
"                out_pairs[2*idx] = i; out_pairs[2*idx+1] = j;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";

static void init_opencl() {
    if (context != NULL) return;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    //clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    program = clCreateProgramWithSource(context, 1, &ocl_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    k_ebond = clCreateKernel(program, "ebond", NULL);
    k_eangle = clCreateKernel(program, "eangle", NULL);
    k_edihedral = clCreateKernel(program, "edihedral", NULL);
    k_enonbonded_cut = clCreateKernel(program, "enonbonded_cut", NULL);
    k_enonbonded_nocut = clCreateKernel(program, "enonbonded_nocut", NULL);
    k_update_nb = clCreateKernel(program, "update_nb", NULL);

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf( "OpenCL: %s\n", device_name );
}

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
    long            i, j, k, *siz, *qms, *fre;

    if( PyArg_ParseTuple( args, "OO", &object, &molecule ) ) {
        init_opencl();
        
        otmp = PyObject_GetAttrString( object, "cut_list" );
        cut = PyFloat_AsDouble( otmp );
        if( cut > 0.0 ) { cut *= cut; } else { cut = 1.0e99; }
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "boxl" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < 3; i++ ) { itm = (double*) PyArray_GETPTR1( mtmp, i ); box[i] = *itm; }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) { itm = (double*) PyArray_GETPTR2( mtmp, i, j ); xyz[k++] = *itm; }
        }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "qmat" );
        qms  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) qms[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "bond" );
        n_bnd = PyList_Size( otmp );
        bnd   = (long*) malloc( max(1, 2*n_bnd) * sizeof( long ) );
        for( i = 0; i < n_bnd; i++ ) {
            bnd[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            bnd[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 1 ) );
        }
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "angl" );
        n_ang = PyList_Size( otmp );
        ang   = (long*) malloc( max(1, 2*n_ang) * sizeof( long ) );
        for( i = 0; i < n_ang; i++ ) {
            ang[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            ang[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 2 ) );
        }
        Py_DECREF( otmp );

        otmp  = PyObject_GetAttrString( object, "dihe" );
        n_dih = PyList_Size( otmp );
        dih   = (long*) malloc( max(1, 2*n_dih) * sizeof( long ) );
        for( i = 0; i < n_dih; i++ ) {
            dih[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            dih[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 3 ) );
        }
        Py_DECREF( otmp );

        // PADDING: Evita que GPUs recorten hilos de la grilla forzando múltiplos de 16
        int max_pairs = siz[0] * 2048;
        cl_mem d_xyz = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * 3 * sizeof(double), xyz, NULL);
        cl_mem d_qms = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(long), qms, NULL);
        cl_mem d_fre = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(long), fre, NULL);
        cl_mem d_bnd = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, max(1, 2*n_bnd) * sizeof(long), bnd, NULL);
        cl_mem d_ang = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, max(1, 2*n_ang) * sizeof(long), ang, NULL);
        cl_mem d_dih = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, max(1, 2*n_dih) * sizeof(long), dih, NULL);
        cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, max_pairs * 2 * sizeof(long), NULL, NULL);
        
        int zero = 0;
        cl_mem d_count = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &zero, NULL);

        int isiz = siz[0], in_bnd = n_bnd, in_ang = n_ang, in_dih = n_dih;
        clSetKernelArg(k_update_nb, 0, sizeof(cl_mem), &d_xyz);
        clSetKernelArg(k_update_nb, 1, sizeof(cl_mem), &d_qms);
        clSetKernelArg(k_update_nb, 2, sizeof(cl_mem), &d_fre);
        clSetKernelArg(k_update_nb, 3, sizeof(cl_mem), &d_bnd);
        clSetKernelArg(k_update_nb, 4, sizeof(cl_mem), &d_ang);
        clSetKernelArg(k_update_nb, 5, sizeof(cl_mem), &d_dih);
        clSetKernelArg(k_update_nb, 6, sizeof(cl_mem), &d_out);
        clSetKernelArg(k_update_nb, 7, sizeof(cl_mem), &d_count);
        clSetKernelArg(k_update_nb, 8, sizeof(int), &isiz);
        clSetKernelArg(k_update_nb, 9, sizeof(double), &box[0]);
        clSetKernelArg(k_update_nb, 10, sizeof(double), &box[1]);
        clSetKernelArg(k_update_nb, 11, sizeof(double), &box[2]);
        clSetKernelArg(k_update_nb, 12, sizeof(double), &cut);
        clSetKernelArg(k_update_nb, 13, sizeof(int), &in_bnd);
        clSetKernelArg(k_update_nb, 14, sizeof(int), &in_ang);
        clSetKernelArg(k_update_nb, 15, sizeof(int), &in_dih);
        clSetKernelArg(k_update_nb, 16, sizeof(int), &max_pairs);

        // Grid Padding Fix: Multiples of 16 prevent thread dropping on OpenCL 1.2 architectures
        size_t local_work_size[2] = { 16, 16 };
        size_t global_work_size[2] = { ((siz[0] + 15) / 16) * 16, ((siz[0] + 15) / 16) * 16 };
        clEnqueueNDRangeKernel(queue, k_update_nb, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

        int count = 0;
        clEnqueueReadBuffer(queue, d_count, CL_TRUE, 0, sizeof(int), &count, 0, NULL, NULL);
        if (count > max_pairs) count = max_pairs;
        
        long* host_out = (long*) malloc(count * 2 * sizeof(long));
        clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, count * 2 * sizeof(long), host_out, 0, NULL, NULL);

        out = PyList_New(count + n_dih);
        long idx = 0;
        for (i = 0; i < count; i++) {
            PyObject *sub = PyList_New(3);
            PyList_SET_ITEM(sub, 0, PyLong_FromLong(host_out[2*i]));
            PyList_SET_ITEM(sub, 1, PyLong_FromLong(host_out[2*i+1]));
            PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(1.0));
            PyList_SET_ITEM(out, idx++, sub);
        }
        for( i = 0; i < n_dih; i++ ) {
            PyObject *sub = PyList_New(3);
            PyList_SET_ITEM(sub, 0, PyLong_FromLong(dih[2*i]));
            PyList_SET_ITEM(sub, 1, PyLong_FromLong(dih[2*i+1]));
            PyList_SET_ITEM(sub, 2, PyFloat_FromDouble(0.5));
            PyList_SET_ITEM(out, idx++, sub);
        }

        clReleaseMemObject(d_xyz); clReleaseMemObject(d_qms); clReleaseMemObject(d_fre);
        clReleaseMemObject(d_bnd); clReleaseMemObject(d_ang); clReleaseMemObject(d_dih);
        clReleaseMemObject(d_out); clReleaseMemObject(d_count);
        free( fre ); free( qms ); free( bnd ); free( ang ); free( dih ); free( xyz ); free(host_out);
        return( out );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_bond( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out_sum = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        init_opencl();
        
        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) { itm = (double*) PyArray_GETPTR2( mtmp, i, j ); xyz[k++] = *itm; }
        }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString( object, "bond" );
        n_lst = PyList_Size( otmp );
        if (n_lst == 0) {
            free(xyz); free(fre); return Py_BuildValue("d", 0.0);
        }
        
        lst   = (long*) malloc( max(1, 2*n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 2; j++ ) lst[2*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp  = PyObject_GetAttrString( object, "bond_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( max(1, 2*n_dat) * sizeof( double ) );
        for( i = 0; i < n_dat; i++ ) 
            for( j = 0; j < 2; j++ ) dat[2*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "bond_indx" );
        ind    = (long*) malloc( max(1, n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        cl_mem d_xyz = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * 3 * sizeof(double), xyz, NULL);
        cl_mem d_lst = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * n_lst * sizeof(long), lst, NULL);
        cl_mem d_dat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * n_dat * sizeof(double), dat, NULL);
        cl_mem d_ind = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_lst * sizeof(long), ind, NULL);
        cl_mem d_fre = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(long), fre, NULL);
        
        cl_mem d_eng_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * sizeof(double), NULL, NULL);
        cl_mem d_grd_out = NULL;
        if (do_grad) d_grd_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * 6 * sizeof(double), NULL, NULL);

        int in_lst = n_lst;
        clSetKernelArg(k_ebond, 0, sizeof(cl_mem), &d_xyz);
        clSetKernelArg(k_ebond, 1, sizeof(cl_mem), &d_lst);
        clSetKernelArg(k_ebond, 2, sizeof(cl_mem), &d_dat);
        clSetKernelArg(k_ebond, 3, sizeof(cl_mem), &d_ind);
        clSetKernelArg(k_ebond, 4, sizeof(cl_mem), &d_fre);
        clSetKernelArg(k_ebond, 5, sizeof(cl_mem), &d_eng_out);
        clSetKernelArg(k_ebond, 6, sizeof(cl_mem), d_grd_out ? &d_grd_out : &d_eng_out);
        clSetKernelArg(k_ebond, 7, sizeof(int), &do_grad);
        clSetKernelArg(k_ebond, 8, sizeof(int), &in_lst);

        size_t local_size = 64;
        size_t global_size = ((n_lst + 63) / 64) * 64;
        clEnqueueNDRangeKernel(queue, k_ebond, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        clEnqueueReadBuffer(queue, d_eng_out, CL_TRUE, 0, n_lst * sizeof(double), host_eng, 0, NULL, NULL);
        
        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 6 * sizeof(double));
            clEnqueueReadBuffer(queue, d_grd_out, CL_TRUE, 0, n_lst * 6 * sizeof(double), host_grd, 0, NULL, NULL);
        }

        otmp = PyObject_GetAttrString( molecule, "grad" );
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );

        for( i = 0; i < n_lst; i++ ) {
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

        if(do_grad) { Py_DECREF( mtmp_grad ); Py_DECREF( otmp ); free(host_grd); }
        free(host_eng);
        clReleaseMemObject(d_xyz); clReleaseMemObject(d_lst); clReleaseMemObject(d_dat);
        clReleaseMemObject(d_ind); clReleaseMemObject(d_fre); clReleaseMemObject(d_eng_out);
        if (d_grd_out) clReleaseMemObject(d_grd_out);
        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre );
        return( Py_BuildValue( "d", out_sum ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_angle( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out_sum = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        init_opencl();
        
        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) { itm = (double*) PyArray_GETPTR2( mtmp, i, j ); xyz[k++] = *itm; }
        }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString( object, "angl" );
        n_lst = PyList_Size( otmp );
        if (n_lst == 0) {
            free(xyz); free(fre); return Py_BuildValue("d", 0.0);
        }

        lst   = (long*) malloc( max(1, 3*n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 3; j++ ) lst[3*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp  = PyObject_GetAttrString( object, "angl_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( max(1, 2*n_dat) * sizeof( double ) );
        for( i = 0; i < n_dat; i++ ) 
            for( j = 0; j < 2; j++ ) dat[2*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "angl_indx" );
        ind    = (long*) malloc( max(1, n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        cl_mem d_xyz = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * 3 * sizeof(double), xyz, NULL);
        cl_mem d_lst = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 3 * n_lst * sizeof(long), lst, NULL);
        cl_mem d_dat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * n_dat * sizeof(double), dat, NULL);
        cl_mem d_ind = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_lst * sizeof(long), ind, NULL);
        cl_mem d_fre = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(long), fre, NULL);
        
        cl_mem d_eng_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * sizeof(double), NULL, NULL);
        cl_mem d_grd_out = NULL;
        if (do_grad) d_grd_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * 9 * sizeof(double), NULL, NULL);

        int in_lst = n_lst;
        clSetKernelArg(k_eangle, 0, sizeof(cl_mem), &d_xyz);
        clSetKernelArg(k_eangle, 1, sizeof(cl_mem), &d_lst);
        clSetKernelArg(k_eangle, 2, sizeof(cl_mem), &d_dat);
        clSetKernelArg(k_eangle, 3, sizeof(cl_mem), &d_ind);
        clSetKernelArg(k_eangle, 4, sizeof(cl_mem), &d_fre);
        clSetKernelArg(k_eangle, 5, sizeof(cl_mem), &d_eng_out);
        clSetKernelArg(k_eangle, 6, sizeof(cl_mem), d_grd_out ? &d_grd_out : &d_eng_out);
        clSetKernelArg(k_eangle, 7, sizeof(int), &do_grad);
        clSetKernelArg(k_eangle, 8, sizeof(int), &in_lst);

        size_t local_size = 64;
        size_t global_size = ((n_lst + 63) / 64) * 64;
        clEnqueueNDRangeKernel(queue, k_eangle, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        clEnqueueReadBuffer(queue, d_eng_out, CL_TRUE, 0, n_lst * sizeof(double), host_eng, 0, NULL, NULL);
        
        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 9 * sizeof(double));
            clEnqueueReadBuffer(queue, d_grd_out, CL_TRUE, 0, n_lst * 9 * sizeof(double), host_grd, 0, NULL, NULL);
        }

        otmp = PyObject_GetAttrString( molecule, "grad" );
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );

        for( i = 0; i < n_lst; i++ ) {
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

        if(do_grad) { Py_DECREF( mtmp_grad ); Py_DECREF( otmp ); free(host_grd); }
        free(host_eng);
        clReleaseMemObject(d_xyz); clReleaseMemObject(d_lst); clReleaseMemObject(d_dat);
        clReleaseMemObject(d_ind); clReleaseMemObject(d_fre); clReleaseMemObject(d_eng_out);
        if (d_grd_out) clReleaseMemObject(d_grd_out);
        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre );
        return( Py_BuildValue( "d", out_sum ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_dihedral( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *itm, *dat;
    long            *siz, i, j, k;
    long            *lst, n_lst, n_dat, *ind, *fre;
    double          out_sum = 0.0;

    if( PyArg_ParseTuple( args, "OOO", &object, &molecule, &gradient ) ) {
        init_opencl();
        
        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) { itm = (double*) PyArray_GETPTR2( mtmp, i, j ); xyz[k++] = *itm; }
        }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "actv" );
        fre  = (long*) malloc( siz[0] * sizeof( long ) );
        for( i = 0; i < siz[0]; i++ ) fre[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp  = PyObject_GetAttrString( object, "dihe" );
        n_lst = PyList_Size( otmp );
        if (n_lst == 0) {
            free(xyz); free(fre); return Py_BuildValue("d", 0.0);
        }

        lst   = (long*) malloc( max(1, 4*n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ )
            for( j = 0; j < 4; j++ ) lst[4*i+j] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
    
        otmp  = PyObject_GetAttrString( object, "dihe_data" );
        n_dat = PyList_Size( otmp );
        dat   = (double*) malloc( max(1, 12*n_dat) * sizeof( double ) );
        for( i = 0; i < 12*n_dat; i++ ) dat[i] = 0.0;
        for( i = 0; i < n_dat; i++ )
            for( j = 0; j < 12; j++ ) dat[12*i+j] = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), j ) );
        Py_DECREF( otmp );
        
        otmp   = PyObject_GetAttrString( object, "dihe_indx" );
        ind    = (long*) malloc( max(1, n_lst) * sizeof( long ) );
        for( i = 0; i < n_lst; i++ ) ind[i] = PyLong_AsLong( PyList_GetItem( otmp, i ) );
        Py_DECREF( otmp );

        cl_mem d_xyz = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * 3 * sizeof(double), xyz, NULL);
        cl_mem d_lst = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * n_lst * sizeof(long), lst, NULL);
        cl_mem d_dat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 12 * n_dat * sizeof(double), dat, NULL);
        cl_mem d_ind = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_lst * sizeof(long), ind, NULL);
        cl_mem d_fre = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(long), fre, NULL);
        
        cl_mem d_eng_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * sizeof(double), NULL, NULL);
        cl_mem d_grd_out = NULL;
        if (do_grad) d_grd_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * 12 * sizeof(double), NULL, NULL);

        int in_lst = n_lst;
        clSetKernelArg(k_edihedral, 0, sizeof(cl_mem), &d_xyz);
        clSetKernelArg(k_edihedral, 1, sizeof(cl_mem), &d_lst);
        clSetKernelArg(k_edihedral, 2, sizeof(cl_mem), &d_dat);
        clSetKernelArg(k_edihedral, 3, sizeof(cl_mem), &d_ind);
        clSetKernelArg(k_edihedral, 4, sizeof(cl_mem), &d_fre);
        clSetKernelArg(k_edihedral, 5, sizeof(cl_mem), &d_eng_out);
        clSetKernelArg(k_edihedral, 6, sizeof(cl_mem), d_grd_out ? &d_grd_out : &d_eng_out);
        clSetKernelArg(k_edihedral, 7, sizeof(int), &do_grad);
        clSetKernelArg(k_edihedral, 8, sizeof(int), &in_lst);

        size_t local_size = 64;
        size_t global_size = ((n_lst + 63) / 64) * 64;
        clEnqueueNDRangeKernel(queue, k_edihedral, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

        double* host_eng = (double*) malloc(n_lst * sizeof(double));
        clEnqueueReadBuffer(queue, d_eng_out, CL_TRUE, 0, n_lst * sizeof(double), host_eng, 0, NULL, NULL);
        
        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 12 * sizeof(double));
            clEnqueueReadBuffer(queue, d_grd_out, CL_TRUE, 0, n_lst * 12 * sizeof(double), host_grd, 0, NULL, NULL);
        }

        otmp = PyObject_GetAttrString( molecule, "grad" );
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );

        for( i = 0; i < n_lst; i++ ) {
            out_sum += host_eng[i];
            if (do_grad) {
                long ai = lst[4*i], aj = lst[4*i+1], ak = lst[4*i+2], al = lst[4*i+3];
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

        if(do_grad) { Py_DECREF( mtmp_grad ); Py_DECREF( otmp ); free(host_grd); }
        free(host_eng);
        clReleaseMemObject(d_xyz); clReleaseMemObject(d_lst); clReleaseMemObject(d_dat);
        clReleaseMemObject(d_ind); clReleaseMemObject(d_fre); clReleaseMemObject(d_eng_out);
        if (d_grd_out) clReleaseMemObject(d_grd_out);
        free( xyz ); free ( lst ); free( dat ); free( ind ); free( fre );
        return( Py_BuildValue( "d", out_sum ) );
    } else { Py_INCREF( Py_None ); return( Py_None ); }
}

// ####################################################################################################################

static PyObject* w_energy_non_bonded( PyObject *self, PyObject *args ) {
    PyObject        *gradient, *object, *molecule, *otmp;
    PyArrayObject   *mtmp;
    double          *xyz, *dat, *scl, *itm, *qms;
    long            *siz, i, j, k;
    long            *lst, n_lst;
    double          oel_sum = 0.0, olj_sum = 0.0, con, cof, box[3], epsi, epsf;

    if( PyArg_ParseTuple( args, "OOOd", &object, &molecule, &gradient, &epsi ) ) {
        init_opencl();
        
        otmp = PyObject_GetAttrString( molecule, "boxl" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < 3; i++ ) { itm = (double*) PyArray_GETPTR1( mtmp, i ); box[i] = *itm; }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( molecule, "coor" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        siz  = PyArray_SHAPE( mtmp );
        xyz  = (double*) malloc( siz[0] * siz[1] * sizeof( double ) );
        for( k = 0, i = 0; i < siz[0]; i++ ) {
            for( j = 0; j < siz[1]; j++ ) { itm = (double*) PyArray_GETPTR2( mtmp, i, j ); xyz[k++] = *itm; }
        }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        otmp = PyObject_GetAttrString( object, "qmat" );
        qms  = (double*) malloc( siz[0] * sizeof( double ) );
        for( i = 0; i < siz[0]; i++ ) qms[i] = PyObject_IsTrue( PyList_GetItem( otmp, i ) ) ? 0.0 : 1.0;
        Py_DECREF( otmp );

        dat  = (double*) malloc( 3 * siz[0] * sizeof( double ) );
        otmp = PyObject_GetAttrString( molecule, "epsi" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) { itm = (double*) PyArray_GETPTR1( mtmp, i ); dat[3*i] = *itm; }
        Py_DECREF( mtmp ); Py_DECREF( otmp );
        
        otmp = PyObject_GetAttrString( molecule, "rmin" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) { itm = (double*) PyArray_GETPTR1( mtmp, i ); dat[3*i+1] = *itm; }
        Py_DECREF( mtmp ); Py_DECREF( otmp );
        
        otmp = PyObject_GetAttrString( molecule, "chrg" );
        mtmp = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );
        for( i = 0; i < siz[0]; i++ ) { itm = (double*) PyArray_GETPTR1( mtmp, i ); dat[3*i+2] = *itm; }
        Py_DECREF( mtmp ); Py_DECREF( otmp );

        int do_grad = (gradient == Py_True) ? 1 : 0;

        otmp = PyObject_GetAttrString( object, "cut_on" ); con = PyFloat_AsDouble( otmp ); Py_DECREF( otmp );
        otmp = PyObject_GetAttrString( object, "cut_off" ); cof = PyFloat_AsDouble( otmp ); Py_DECREF( otmp );
    
        otmp  = PyObject_GetAttrString( object, "nbnd" );
        n_lst = PyList_Size( otmp );
        if (n_lst == 0) {
            free( xyz ); free( dat ); free( qms ); return Py_BuildValue( "(d,d)", 0.0, 0.0 );
        }

        lst   = (long*) malloc( max(1, 2*n_lst) * sizeof( long ) );
        scl   = (double*) malloc( max(1, n_lst) * sizeof( double ) );
        for( i = 0; i < n_lst; i++ ) {
            lst[2*i]   = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 0 ) );
            lst[2*i+1] = PyLong_AsLong( PyList_GetItem( PyList_GetItem( otmp, i ), 1 ) );
            scl[i]     = PyFloat_AsDouble( PyList_GetItem( PyList_GetItem( otmp, i ), 2 ) );
        }
        Py_DECREF( otmp );

        epsf = 1389.35484620709144110151 / epsi;

        cl_mem d_xyz = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * 3 * sizeof(double), xyz, NULL);
        cl_mem d_lst = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * n_lst * sizeof(long), lst, NULL);
        cl_mem d_dat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 3 * siz[0] * sizeof(double), dat, NULL);
        cl_mem d_qms = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, siz[0] * sizeof(double), qms, NULL);
        cl_mem d_scl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_lst * sizeof(double), scl, NULL);

        cl_mem d_oel_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * sizeof(double), NULL, NULL);
        cl_mem d_olj_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * sizeof(double), NULL, NULL);
        cl_mem d_grd_out = NULL;
        if (do_grad) d_grd_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_lst * 6 * sizeof(double), NULL, NULL);

        int in_lst = n_lst;
        size_t local_size = 64;
        size_t global_size = ((n_lst + 63) / 64) * 64; // Grid Padding 1D

        if( con > 0.0 && cof > con ) {
            double c2on = con * con, c2of = cof * cof;
            double _g   = pow( c2of - c2on, 3.0 );
            double _a   = c2of * c2of * ( c2of - 3.0 * c2on ) / _g;
            double _b   = 6.0 * c2of * c2on / _g;
            double _c   = - ( c2of + c2on ) / _g;
            double _d   = 0.4 / _g;
            double _el1 = 8.0 * ( c2of * c2on * ( cof - con ) - 0.2 * ( cof * c2of * c2of - con * c2on * c2on ) ) / _g;
            double _el2 = - _a / cof + _b * cof + _c * cof * c2of + _d * cof * c2of * c2of;
            double k6   = ( cof * c2of ) / ( cof * c2of - con * c2on );
            double k12  = pow( c2of, 3.0 ) / ( pow( c2of, 3.0 ) - pow( c2on, 3.0 ) );

            clSetKernelArg(k_enonbonded_cut, 0, sizeof(cl_mem), &d_xyz);
            clSetKernelArg(k_enonbonded_cut, 1, sizeof(cl_mem), &d_lst);
            clSetKernelArg(k_enonbonded_cut, 2, sizeof(cl_mem), &d_dat);
            clSetKernelArg(k_enonbonded_cut, 3, sizeof(cl_mem), &d_qms);
            clSetKernelArg(k_enonbonded_cut, 4, sizeof(cl_mem), &d_scl);
            clSetKernelArg(k_enonbonded_cut, 5, sizeof(cl_mem), &d_oel_out);
            clSetKernelArg(k_enonbonded_cut, 6, sizeof(cl_mem), &d_olj_out);
            clSetKernelArg(k_enonbonded_cut, 7, sizeof(cl_mem), do_grad ? &d_grd_out : &d_oel_out);
            clSetKernelArg(k_enonbonded_cut, 8, sizeof(double), &box[0]);
            clSetKernelArg(k_enonbonded_cut, 9, sizeof(double), &box[1]);
            clSetKernelArg(k_enonbonded_cut, 10, sizeof(double), &box[2]);
            clSetKernelArg(k_enonbonded_cut, 11, sizeof(double), &epsf);
            clSetKernelArg(k_enonbonded_cut, 12, sizeof(double), &c2on);
            clSetKernelArg(k_enonbonded_cut, 13, sizeof(double), &c2of);
            clSetKernelArg(k_enonbonded_cut, 14, sizeof(double), &con);
            clSetKernelArg(k_enonbonded_cut, 15, sizeof(double), &cof);
            clSetKernelArg(k_enonbonded_cut, 16, sizeof(double), &_a);
            clSetKernelArg(k_enonbonded_cut, 17, sizeof(double), &_b);
            clSetKernelArg(k_enonbonded_cut, 18, sizeof(double), &_c);
            clSetKernelArg(k_enonbonded_cut, 19, sizeof(double), &_d);
            clSetKernelArg(k_enonbonded_cut, 20, sizeof(double), &_el1);
            clSetKernelArg(k_enonbonded_cut, 21, sizeof(double), &_el2);
            clSetKernelArg(k_enonbonded_cut, 22, sizeof(double), &k6);
            clSetKernelArg(k_enonbonded_cut, 23, sizeof(double), &k12);
            clSetKernelArg(k_enonbonded_cut, 24, sizeof(int), &do_grad);
            clSetKernelArg(k_enonbonded_cut, 25, sizeof(int), &in_lst);

            clEnqueueNDRangeKernel(queue, k_enonbonded_cut, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

        } else {
            clSetKernelArg(k_enonbonded_nocut, 0, sizeof(cl_mem), &d_xyz);
            clSetKernelArg(k_enonbonded_nocut, 1, sizeof(cl_mem), &d_lst);
            clSetKernelArg(k_enonbonded_nocut, 2, sizeof(cl_mem), &d_dat);
            clSetKernelArg(k_enonbonded_nocut, 3, sizeof(cl_mem), &d_qms);
            clSetKernelArg(k_enonbonded_nocut, 4, sizeof(cl_mem), &d_scl);
            clSetKernelArg(k_enonbonded_nocut, 5, sizeof(cl_mem), &d_oel_out);
            clSetKernelArg(k_enonbonded_nocut, 6, sizeof(cl_mem), &d_olj_out);
            clSetKernelArg(k_enonbonded_nocut, 7, sizeof(cl_mem), do_grad ? &d_grd_out : &d_oel_out);
            clSetKernelArg(k_enonbonded_nocut, 8, sizeof(double), &box[0]);
            clSetKernelArg(k_enonbonded_nocut, 9, sizeof(double), &box[1]);
            clSetKernelArg(k_enonbonded_nocut, 10, sizeof(double), &box[2]);
            clSetKernelArg(k_enonbonded_nocut, 11, sizeof(double), &epsf);
            clSetKernelArg(k_enonbonded_nocut, 12, sizeof(int), &do_grad);
            clSetKernelArg(k_enonbonded_nocut, 13, sizeof(int), &in_lst);

            clEnqueueNDRangeKernel(queue, k_enonbonded_nocut, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        }

        double* host_oel = (double*) malloc(n_lst * sizeof(double));
        double* host_olj = (double*) malloc(n_lst * sizeof(double));
        clEnqueueReadBuffer(queue, d_oel_out, CL_TRUE, 0, n_lst * sizeof(double), host_oel, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_olj_out, CL_TRUE, 0, n_lst * sizeof(double), host_olj, 0, NULL, NULL);
        
        double* host_grd = NULL;
        if (do_grad) {
            host_grd = (double*) malloc(n_lst * 6 * sizeof(double));
            clEnqueueReadBuffer(queue, d_grd_out, CL_TRUE, 0, n_lst * 6 * sizeof(double), host_grd, 0, NULL, NULL);
        }

        otmp = PyObject_GetAttrString( molecule, "grad" );
        PyArrayObject* mtmp_grad = NULL;
        if (do_grad) mtmp_grad = (PyArrayObject*) PyArray_FROM_OT( otmp, NPY_DOUBLE );

        for( i = 0; i < n_lst; i++ ) {
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

        if(do_grad) { Py_DECREF( mtmp_grad ); Py_DECREF( otmp ); free(host_grd); }
        free(host_oel); free(host_olj);
        clReleaseMemObject(d_xyz); clReleaseMemObject(d_lst); clReleaseMemObject(d_dat);
        clReleaseMemObject(d_qms); clReleaseMemObject(d_scl);
        clReleaseMemObject(d_oel_out); clReleaseMemObject(d_olj_out);
        if (d_grd_out) clReleaseMemObject(d_grd_out);
        free( xyz ); free( dat ); free ( lst ); free( scl ); free( qms );
        return( Py_BuildValue( "(d,d)", oel_sum, olj_sum ) );
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
