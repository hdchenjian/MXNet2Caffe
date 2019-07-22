from predictor_caffe import PredictorCaffe
import numpy as np
import mxnet as mx

from collections import namedtuple

Batch = namedtuple('Batch',['data'])

size = (1, 3, 112, 112)
netcaffe = PredictorCaffe('model_caffe/face_30/face_30.prototxt', 'model_caffe/face_30/face_30.caffemodel', size)

def mx_model(prefix_mxnet, size):
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
    return
    
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
    
def caffe_model(prototxt, caffemodel, size):
    tensor = np.ones(size, dtype=np.float32)
    img = (tensor-127.5) / 128
    netcaffe.forward(img)
    out_caffe = netcaffe.blob_by_name("fc1")
    out_caffe_data = out_caffe.data.tolist()
    print out_caffe_data[0][0:10]

    
if __name__ == "__main__":
    prefix_mxnet = "/var/darknet/insightface/models/model-r34-amf/model"
    size = (1, 3, 112, 112)
    mx_model(prefix_mxnet, size)
    caffe_model('model_caffe/face_30/face_30.prototxt', 'model_caffe/face_30/face_30.caffemodel', size)
