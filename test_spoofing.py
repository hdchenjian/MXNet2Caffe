import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "/home/luyao/git/caffe/python"))
import caffe
import numpy as np

net = caffe.Net('model_caffe/spoofing_residual/spoofing_cnn.prototxt', 'model_caffe/spoofing_residual/spoofing_cnn.caffemodel', caffe.TEST)

    
def caffe_model():
    img = np.ones((1, 3, 112, 96), dtype=np.float32)
    img = img / 2
    net.blobs['data'].data[...] = img
    net.forward()
    blob_name = 'conv00'
    print(blob_name, net.blobs[blob_name].data)
    blob_name = 'relu0'
    print(blob_name, net.blobs[blob_name].data[0][0][0][0:10])
    blob_name = 'pool1'
    print(blob_name, net.blobs[blob_name].data[0][0:5])
    print(len( net.params['conv00'][0].data.flat), net.params['conv00'][0].data.flat[0:10])
    print(len( net.params['conv00'][1].data.flat), net.params['conv00'][1].data.flat[0:10])


if __name__ == "__main__":
    caffe_model()
