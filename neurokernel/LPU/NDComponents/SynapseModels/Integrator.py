from BaseSynapseModel import BaseSynapseModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron
cuda_src = """


__global__ void integrator(
    int num,
    %(type)s dt,
    %(type)s *g_I,
    %(type)s *gamma,
    %(type)s *internal_var,
    %(type)s *out)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    for( int i=tid; i<num; i+=tot_threads ){
        internal_var[i] = internal_var[i] - dt * gamma[i] * internal_var[i];
        internal_var[i] += dt * g_I[i];
        internal_var[i] = max(-0.,min(1., internal_var[i]));
        out[i] = internal_var[i];
    }
    return;
}
"""
class Integrator(BaseSynapseModel):
    accesses = ['I']
    updates = ['I']
    def __init__( self, params_dict, access_buffers, dt,
                  LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num = params_dict['gamma'].size
        self.LPU_id = LPU_id

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.inputs = {}
        self.inputs['I'] = garray.zeros( (self.num), dtype=np.float64 )
        self.internal_var   = garray.zeros( (self.num,), dtype=np.float64 )
        print(self.accesses)

        self.update = self.get_gpu_kernel(params_dict['gamma'].dtype)


    def run_step(self, update_pointers, st = None):
        self.sum_in_variable('I', self.inputs['I'], st=st)

        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.inputs['I'].gpudata,\
            self.params_dict['gamma'].gpudata,\
            self.internal_var.gpudata,\
            update_pointers['I'])

    def get_gpu_kernel(self, dtype=np.double):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(dtype)},\
                            options=self.compile_options)
        func = mod.get_function("integrator")
        func.prepare('idPPPP')
        return func
