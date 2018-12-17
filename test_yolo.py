import numpy as np
import struct
try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/home/luyao/git/caffe/python"))
    import caffe

def compare_models(prefix_caffe):
    net = caffe.Net(prefix_caffe + ".prototxt", prefix_caffe + ".caffemodel", caffe.TEST)
    img = np.ones((1, 3, 416, 416), dtype=np.float32)
    img = img / 2
    print('welll\n')
    net.blobs['data'].data[...] = img
    net.forward()
    print(type(net.blobs['conv21'].data[0]))
    print(type(net.blobs['conv21'].data), net.blobs['conv21'].data.shape)
    for i in range(1, 28):
        blob_name = 'conv' + str(i)
        if blob_name in net.blobs:
            print(blob_name, net.blobs[blob_name].data[0][0][0][0:10])
            print('')
        blob_name = 'bn' + str(i)
        if blob_name in net.blobs:
            print(blob_name, net.blobs[blob_name].data[0][0][0][0:10])
            print('')
        blob_name = 'relu' + str(i)
        if blob_name in net.blobs:
            print(blob_name, net.blobs[blob_name].data[0][0][0][0:10])
            print('')

    print('conv21', net.blobs['conv21'].data[0][0][0])
    print len(net.params['conv1']), len(net.params['conv21'])
    return
    print(net.blobs['conv21'].data[0][0][0])
    print(type(net.blobs['conv1'].data), net.blobs['conv1'].data.shape)
    print(net.blobs['data'].data[0][0][0][0:10])
    print net.params['conv1'][0].data.flat[0:10]
    print('conv1', net.blobs['conv1'].data[0][0][0][0:10])
    print('bn1', net.blobs['bn1'].data[0][0][0][0:10])
    print('relu1', net.blobs['relu1'].data[0][0][0][0:10])

    print('mean', net.params['bn1'][0].data.flat[0:10])
    print('var', net.params['bn1'][1].data.flat[0:10])
    print(net.params['bn1_scale'][0].data.flat[0:10])
    print(net.params['bn1_scale'][1].data.flat[0:10])
    
    
if __name__ == "__main__":
    prefix_caffe = "model_caffe/yolo-tiny-more/yolov3-tiny-more_split_conv_layer"
    compare_models(prefix_caffe)
