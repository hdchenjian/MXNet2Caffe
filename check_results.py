from predictor_caffe import PredictorCaffe
from predictor_mxnet import PredictorMxNet
import numpy as np

def compare_diff_sum(tensor1, tensor2):
    pass

def compare_cosin_dist(tensor1, tensor2):
    pass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compare_models(prefix_mxnet, prefix_caffe, size):
    netmx = PredictorMxNet(prefix_mxnet, 0, size)    
    model_file = prefix_caffe + ".prototxt"
    pretrained_file = prefix_caffe + ".caffemodel"
    netcaffe = PredictorCaffe(model_file, pretrained_file, size)
    tensor = np.ones(size, dtype=np.float32)
    out_mx = netmx.forward(tensor)
    out_mx = out_mx[0].asnumpy()
    #print out_mx
    img = (tensor-127.5) / 128
    netcaffe.forward(img)
    out_caffe = netcaffe.blob_by_name("fc1")
    #print out_caffe.data
    out_caffe_data = out_caffe.data.tolist()
    #print(len(out_caffe_data[0]), len(out_mx[0][0]), type(out_caffe_data), type(out_mx))
    for i in range(0, 10):
        #print(out_mx[0][0][i], out_caffe_data[0][i])
        print(out_mx[0][i], out_caffe_data[0][i])
    print(type(netcaffe.net.blobs))
    #print(netcaffe.net.blobs.keys())
    blob_name = 'stage1_unit1_relu1' # 2
    blob_name = 'stage1_unit1_bn3' # 3
    #blob_name = 'stage1_unit1_conv2'
    blob_name = 'stage1_unit1_sc'
    #blob_name = 'stage1_unit1_conv1sc'
    #blob_name = 'relu0'
    blob_name = 'stage1_unit2_relu1' #8
    blob_name = 'stage4_unit3_bn3' #103
    blob_name = 'bn1' # 105
    #blob_name = 'pre_fc1'
    blob_name = 'stage1_unit1_bn3'
    out_caffe = netcaffe.blob_by_name(blob_name)
    print(type(out_caffe.data), type(out_caffe.data.tolist()), len(out_caffe.data.flat))
    count = 0
    index = 0
    for element in out_caffe.data.flat:
        if count >= 10: break
        if element > 0.001:
            print(index, element)
            count += 1
        index += 1
    '''
    for i in range(0, len(netcaffe.net.params[blob_name][0].data.flat)):
        print(i, netcaffe.net.params[blob_name][0].data.flat[i], netcaffe.net.params[blob_name][1].data.flat[i],
              netcaffe.net.params[blob_name + '_scale'][0].data.flat[i], netcaffe.net.params[blob_name + '_scale'][1].data.flat[i])
    index = 0
    for _w in netcaffe.net.params['conv0'][0].data.flat:
        if index < 10: print(index, _w)
        else: break
        index += 1
    '''
    sum0 = 0
    sum1 = 0
    for i in range(0, len(out_mx[0])):
        #print(out_mx[0][0][i], out_caffe_data[0][i])
        sum0 += out_mx[0][i] * out_mx[0][i]
        sum1 += out_caffe_data[0][i] * out_caffe_data[0][i]
    #print softmax(out_caffe.data)
    #print softmax(out_caffe.data)
    print(sum0, sum1);
    print "done"
    
if __name__ == "__main__":
    prefix_mxnet = "/var/darknet/insightface/models/model-r50-am-lfw/model"
    prefix_caffe = "model_caffe/face/facega2"
    size = (1, 3, 112, 112)
    compare_models(prefix_mxnet, prefix_caffe, size)
