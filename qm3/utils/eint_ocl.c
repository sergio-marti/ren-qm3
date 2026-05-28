#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Kernel OpenCL incrustado como string.
// Usa rsqrt() para calcular 1/r a nivel de hardware en un solo ciclo de reloj.
const char *kernel_source = 
"__kernel void calc_elec_pot(__global const double* g_coor,   \n"
"                            __global const double* g_chrg,   \n"
"                            __global const double* mm_coor,  \n"
"                            __global const double* mm_chrg,  \n"
"                            __global double* out_energy,     \n"
"                            const int n_mm) {                \n"
"    int i = get_global_id(0);                                \n"
"    double gx = g_coor[i*3 + 0];                             \n"
"    double gy = g_coor[i*3 + 1];                             \n"
"    double gz = g_coor[i*3 + 2];                             \n"
"    double pot = 0.0;                                        \n"
"                                                             \n"
"    for(int j = 0; j < n_mm; j++) {                          \n"
"        double dx = gx - mm_coor[j*3 + 0];                   \n"
"        double dy = gy - mm_coor[j*3 + 1];                   \n"
"        double dz = gz - mm_coor[j*3 + 2];                   \n"
"        double r2 = dx*dx + dy*dy + dz*dz;                   \n"
"        if (r2 < 1e-10) r2 = 1e-10;                          \n"
"        pot += mm_chrg[j] * rsqrt(r2);                       \n"
"    }                                                        \n"
"    out_energy[i] = g_chrg[i] * pot;                         \n"
"}                                                            \n";

static PyObject* w_calc_rho_elec_gpu(PyObject* self, PyObject* args) {
    PyArrayObject *py_g_coor, *py_g_chrg, *py_mm_coor, *py_mm_chrg;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
            &PyArray_Type, &py_g_coor, 
            &PyArray_Type, &py_g_chrg, 
            &PyArray_Type, &py_mm_coor, 
            &PyArray_Type, &py_mm_chrg)) {
        return NULL;
    }

    int n_grid = (int)PyArray_DIM(py_g_chrg, 0);
    int n_mm   = (int)PyArray_DIM(py_mm_chrg, 0);

    double *g_coor_ptr = (double*)PyArray_DATA(py_g_coor);
    double *g_chrg_ptr = (double*)PyArray_DATA(py_g_chrg);
    double *mm_coor_ptr = (double*)PyArray_DATA(py_mm_coor);
    double *mm_chrg_ptr = (double*)PyArray_DATA(py_mm_chrg);

    double *out_energy = (double*)malloc(n_grid * sizeof(double));

    // -----------------------------------------------------------------------------------------
    // CONFIGURACIÓN ESTÁNDAR DE OPENCL
    // -----------------------------------------------------------------------------------------
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "calc_elec_pot", NULL);

    cl_mem b_g_coor = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_grid * 3 * sizeof(double), g_coor_ptr, NULL);
    cl_mem b_g_chrg = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_grid * sizeof(double), g_chrg_ptr, NULL);
    cl_mem b_mm_coor = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_mm * 3 * sizeof(double), mm_coor_ptr, NULL);
    cl_mem b_mm_chrg = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_mm * sizeof(double), mm_chrg_ptr, NULL);
    cl_mem b_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_grid * sizeof(double), NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_g_coor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_g_chrg);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_mm_coor);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_mm_chrg);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_out);
    clSetKernelArg(kernel, 5, sizeof(int), &n_mm);

    size_t global_size = n_grid;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, b_out, CL_TRUE, 0, n_grid * sizeof(double), out_energy, 0, NULL, NULL);

    double total_e_ele = 0.0;
    for(int i = 0; i < n_grid; i++) { total_e_ele += out_energy[i]; }

    clReleaseMemObject(b_g_coor); clReleaseMemObject(b_g_chrg);
    clReleaseMemObject(b_mm_coor); clReleaseMemObject(b_mm_chrg);
    clReleaseMemObject(b_out);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(out_energy);

    return Py_BuildValue("d", total_e_ele);
}

static struct PyMethodDef methods[] = {
    {"calc_rho_elec", (PyCFunction)w_calc_rho_elec_gpu, METH_VARARGS, "Calcula la energia electrostatica del grid en GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mod_definition = {
    PyModuleDef_HEAD_INIT,
    "_eint",
    "OpenCL rho electrostatic interaction (cube)",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__eint(void) {
    import_array();
    return( PyModule_Create(&mod_definition) );
}
