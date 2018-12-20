import cv2
import numpy as np
import struct
try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/home/luyao/git/caffe/python"))
    import caffe

def compare_models():
    net = caffe.Net('/home/luyao/git/MXNet2Caffe/model_caffe/darknet_mtcnn/det3_permute.prototxt',
                    '/home/luyao/git/MXNet2Caffe/model_caffe/darknet_mtcnn/det3_permute.caffemodel', caffe.TEST)
    net1 = caffe.Net('/home/luyao/git/MXNet2Caffe/model_caffe/det3_permute.prototxt',
                    '/home/luyao/git/MXNet2Caffe/model_caffe/det3_bgr.caffemodel', caffe.TEST)
    tensor = np.ones((1, 3, 48, 48), dtype=np.float32)
    img = tensor / 2
    net1.blobs['data'].data[...] = img
    net1.forward()
    net.blobs['data'].data[...] = img
    net.forward()
    print('\n\n\n')
    print('conv1', net1.params['conv1'][0].data.shape, sum(net.params['conv1'][0].data != net1.params['conv1'][0].data))
    print('conv1 bias', net1.params['conv1'][1].data.shape, net.params['conv1'][1].data != net1.params['conv1'][1].data)
    #print('relu1', net1.params['conv1'][0].data.shape, sum(net.params['conv1'][0].data != net1.params['conv1'][0].data))
    key_i = 'prelu1'
    print(key_i, type(net.params[key_i][0].data), net.params[key_i][0].data.shape)
    print('prelu1', net1.params[key_i][0].data.shape, net.params[key_i][0].data != net1.params[key_i][0].data)
    print('conv6-3', net.blobs['conv6-3'].data.shape, net.blobs['conv6-3'].data != net1.blobs['conv6-3'].data)
    print(net.blobs['conv6-3'].data)
    print(net1.blobs['conv6-3'].data)
    print(net.blobs['conv4'].data[0][0])
    #net.params[key_i][0].data.flat = [0.1] * get_weight_num(net.params[key_i][0].data.shape)
    #print('conv1 diff', img.shape, img != img)
    #print('conv1 diff', net.blobs['conv1'].data.shape, sum(net.blobs['conv1'].data - net1.blobs['conv1'].data))
    #print('conv1', net.blobs['conv1'].data[0][0][0])
    #print('conv1 right', net1.blobs['conv1'].data[0][0][0])
    #print(net.blobs['conv1'].data - net1.blobs['conv1'].data)
    #print(net.params['conv2'][0].data == net1.params['conv2'][0].data)
    #print(net.params['conv2'][0].data == net1.params['conv2'][0].data)
    #print('conv1', net.blobs['conv1'].data.shape, sum(sum(sum(net.blobs['conv1'].data != net.blobs['conv1'].data))))
    #print('conv2', net.blobs['conv2'].data.shape, sum(sum(sum(net.blobs['conv2'].data != net.blobs['conv2'].data))))
    #print('conv5', net.blobs['conv5'].data.shape, net.blobs['conv5'].data != net.blobs['conv5'].data)
    #print('conv6-3', net.blobs['conv6-3'].data.shape, net.blobs['conv6-3'].data != net.blobs['conv6-3'].data)
    
    
if __name__ == "__main__":
    compare_models()
