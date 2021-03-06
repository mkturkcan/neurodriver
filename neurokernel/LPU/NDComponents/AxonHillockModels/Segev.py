from BaseAxonHillockModel import BaseAxonHillockModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron
cuda_src = """


__global__ void segev(
    int num,
    %(type)s dt,
    %(type)s *g_I,
    %(type)s *C,
    %(type)s *R,
    %(type)s *V_leak,
    %(type)s *internal_var,
    %(type)s *spike_state,
    %(type)s *out)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    for( int i=tid; i<num; i+=tot_threads ){
        internal_var[i] = internal_var[i] + dt * ( (V_leak[i] - internal_var[i]) / R[i] / C[i] ) +  dt * g_I[i];
        out[i] = internal_var[i];
        spike_state[i] = 0;
    }
    return;
}
"""
class Segev(BaseAxonHillockModel):
    accesses = ['I']
    updates = ['V', 'spike_state']
    def __init__( self, params_dict, access_buffers, dt,
                  LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num = params_dict['C'].size
        self.LPU_id = LPU_id

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.inputs = {}
        self.inputs['I'] = garray.zeros( (self.num), dtype=np.float64 )
        self.internal_var   = garray.zeros( (self.num,), dtype=np.float64 )
        print(self.accesses)

        self.update = self.get_gpu_kernel(params_dict['C'].dtype)


    def run_step(self, update_pointers, st = None):
        self.sum_in_variable('I', self.inputs['I'], st=st)

        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.inputs['I'].gpudata,\
            self.params_dict['C'].gpudata,\
            self.params_dict['R'].gpudata,\
            self.params_dict['V_leak'].gpudata,\
            self.internal_var.gpudata,\
            update_pointers['spike_state'],
            update_pointers['V'])

    def get_gpu_kernel(self, dtype=np.double):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(dtype)},\
                            options=self.compile_options)
        func = mod.get_function("segev")
        func.prepare('idPPPPPPP')
        return func
