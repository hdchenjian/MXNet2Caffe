from predictor_mxnet import PredictorMxNet
import numpy as np
import mxnet as mx

from collections import namedtuple

Batch = namedtuple('Batch',['data'])

def compare_models(prefix_mxnet, size):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix_mxnet, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data',size)], label_shapes = mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    tensor = np.ones(size, dtype=np.float32)
    #print tensor[0:10]
    mod.forward(Batch([mx.nd.array(tensor)]))
    out_mx = mod.get_outputs()
    out_mx = out_mx[0].asnumpy()
    print out_mx[0][0:10], '\n\n'
    
    #aa = netmx.mod.get_internals()
    internals = mod.symbol.get_internals()
    #print internals.list_outputs()
    tmp_layer = internals['conv0_output']
    mod3 = mx.mod.Module(symbol=tmp_layer, context=mx.gpu(), label_names=None)
    mod3.bind(for_training=False, data_shapes=[('data', size)])
    mod3.set_params(arg_params, aux_params)
    mod3.forward(Batch([mx.nd.array(tensor)]))
    out_mx = mod3.get_outputs()
    out_mx = out_mx[0].asnumpy()
    print out_mx[0][0][0][0:10]
    print out_mx.shape
    
    
    
if __name__ == "__main__":
    prefix_mxnet = "/var/darknet/insightface/models/model-r34-amf/model"
    size = (1, 3, 112, 112)
    compare_models(prefix_mxnet, size)
