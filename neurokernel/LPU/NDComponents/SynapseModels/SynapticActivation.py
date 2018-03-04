from BaseSynapseModel import BaseSynapseModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron
cuda_src = """


__global__ void chemical(
    int num,
    %(type)s dt,
    %(type)s *g_V,
    %(type)s *g_max,
    %(type)s *K,
    %(type)s *V_eq,
    %(type)s *V_range,
    %(type)s *out)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    for( int i=tid; i<num; i+=tot_threads ){
        out[i] = 0. * g_max[i] / (1.0 + expf(K[i] * 0.00001 * (g_V[i] - V_eq[i]) / V_range[i]));
    }
    return;
}
"""

class Chemical(BaseSynapseModel):
    accesses = ['V']
    updates = ['g']
    def __init__( self, params_dict, access_buffers, dt,
                  LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num = params_dict['g_max'].size
        self.LPU_id = LPU_id

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.inputs = {}
        self.inputs['V'] = garray.zeros( (self.num), dtype=np.float64 )
        print(self.accesses)

        self.update = self.get_gpu_kernel(params_dict['g_max'].dtype)


    def run_step(self, update_pointers, st = None):
        self.sum_in_variable('V', self.inputs['V'], st=st)
        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.inputs['V'].gpudata,\
            self.params_dict['g_max'].gpudata,\
            self.params_dict['K'].gpudata,\
            self.params_dict['V_eq'].gpudata,\
            self.params_dict['V_range'].gpudata,\
            update_pointers['g'])

    def get_gpu_kernel(self, dtype=np.double):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(dtype)},\
                            options=self.compile_options)
        func = mod.get_function("chemical")
        func.prepare('idPPPPPP')
        return func
