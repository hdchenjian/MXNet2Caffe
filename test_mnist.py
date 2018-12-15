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

def compare_models(prefix_caffe):
    net = caffe.Net(prefix_caffe + ".prototxt", prefix_caffe + ".caffemodel", caffe.TEST)
    with open('/home/luyao/git/caffe/data/mnist/t10k-images-idx3-ubyte' ,'rb') as f:
        buf1 = f.read()
    image_index = 0
    image_index += struct.calcsize('>IIII')
    temp = struct.unpack_from('>784B', buf1, image_index)
    img = np.reshape(temp,(28,28)).astype(np.uint8)
    #print(img)
    print(img.shape, type(img[0][0]))
    #img = cv2.merge([img, img , img])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('image',img)
    cv2.imwrite('image.png', img)
    print(img[0])
    print(img[7])
    img1 = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    print('read img', img1.shape, sum(sum((img1 == img) != True)))
    print(img1[0])
    print(img1[7])

    #cv2.waitKey(0)

    img = img * 0.00390625
    net.blobs['data'].data[...] = img
    #print(img)
    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]):
            #pass
            sys.stdout.write(str(img[i][j]) + " ")
        print()
        print(i+1)
    net.forward()
    print(type(net.blobs['ip2'].data[0]))
    print(net.blobs['ip2'].data[0])
    print(net.blobs['loss'].data[0])
    print(type(net.blobs['conv1'].data), net.blobs['conv1'].data.shape)
    print(net.blobs['conv1'].data[0][0][0])
    print(net.blobs['conv1'].data[0][1][0])

    
    
if __name__ == "__main__":
    prefix_caffe = "/home/luyao/git/caffe/examples/mnist/lenet_iter_10000"
    compare_models(prefix_caffe)
