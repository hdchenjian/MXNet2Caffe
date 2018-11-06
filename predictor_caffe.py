import numpy as np
try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/opt/ego/caffe-rcnn-face-ssd/python"))
    import caffe

class PredictorCaffe:
    def __init__(self, model_file, pretrained_file, size):
        self.net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        self.size = size
        
    def forward(self, tensor, data="data"):
        self.net.blobs[data].data[...] = tensor
        self.net.forward()
        
    def blob_by_name(self, blobname):
        return self.net.blobs[blobname]
