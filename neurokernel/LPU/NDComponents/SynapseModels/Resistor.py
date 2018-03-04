from BaseSynapseModel import BaseSynapseModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron
cuda_src = """


__global__ void resistor(
    int num,
    %(type)s dt,
    %(type)s *V,
    int ld,
    int current,
    int buffer_length,
    int *Pre,
    int *npre,
    int *cumpre,
    int *delay,
    %(type)s *tag,
    %(type)s *R,
    %(type)s *out)
{
    int V1, V2;

    int col;
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    for( int i=tid; i<num; i+=tot_threads ){
    if (npre[i]==2)
    {
    col = current-delay[i];
    if(col < 0)
    {
        col = buffer_length + col;
    }
    //pre = col*ld + Pre[cumpre[i]];
    V1 =  col*ld + Pre[cumpre[i]];
    V2 =  col*ld + Pre[cumpre[i]+1];







    if (tag[cumpre[i]])
        {out[i] = 1.0 / R[i] * (V[V1] - V[V2]);}
    else
        {out[i] = 1.0 / R[i]  * (V[V2] - V[V1]);}
        }
    }
    return;
}
"""
class Resistor(BaseSynapseModel):
    accesses = ['V']
    updates = ['I']
    def __init__( self, params_dict, access_buffers, dt,
                  LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num = params_dict['R'].size
        self.LPU_id = LPU_id

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        #self.inputs = {}
        #self.inputs['I'] = garray.zeros( (self.num), dtype=np.float64 )
        print(self.accesses)

        self.update = self.get_gpu_kernel(params_dict['R'].dtype)


    def run_step(self, update_pointers, st = None):
        #self.sum_in_variable('V', self.inputs['V'], st=st)

        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.access_buffers['V'].gpudata,
            self.access_buffers['V'].ld,
            self.access_buffers['V'].current,
            self.access_buffers['V'].buffer_length,
            self.params_dict['pre']['V'].gpudata,
            self.params_dict['npre']['V'].gpudata,
            self.params_dict['cumpre']['V'].gpudata,
            self.params_dict['conn_data']['V']['delay'].gpudata,
            self.params_dict['conn_data']['V']['tag'].gpudata,
            self.params_dict['R'].gpudata,
            update_pointers['I']
            )

    def get_gpu_kernel(self, dtype=np.double):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(dtype)},\
                            options=self.compile_options)
        func = mod.get_function("resistor")
        func.prepare('idPiiiPPPPPPP')
        return func
