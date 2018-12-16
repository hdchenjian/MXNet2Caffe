import sys, argparse
try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/home/luyao/git/caffe/python"))
    import caffe

import traceback
import struct
from array import array

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
#parser.add_argument('--mx-model',        type=str, default='/var/darknet/insightface/models/model-r50-am-lfw/model')
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/yolo-tiny-more/yolov3-tiny-more.prototxt')
parser.add_argument('--cf-model',        type=str, default='model_caffe/yolo-tiny-more/yolov3-tiny-more.caffemodel')
args = parser.parse_args()

net = caffe.Net(args.cf_prototxt, caffe.TRAIN)
network_keys = []
f = open(args.cf_prototxt, 'rU')
for line in f.readlines():
    if 'name: "' not in line:
        continue
    if 'mxnet-' in line: continue
    if 'bn' not in line and 'conv' not in line:
        continue
    line = line[line.find('name: ') + 7 : -2]
    print(line)
    network_keys.append(line)
f.close()
#print(network_keys)

weight_file = open('/home/luyao/download/Snapdragon/snpe-1.21.0/examples/NativeCpp/yolov3-tiny-more_final.weights', 'rb')
darknet_weight = weight_file.read()
weight_file.close()
print(struct.unpack("iiiii", darknet_weight[:20]), len(darknet_weight), (len(darknet_weight) - 20) / 4.0)
darknet_weight = struct.unpack("f" * ((len(darknet_weight) - 20) / 4), darknet_weight[20:])
print(len(darknet_weight))

def get_weight_num(a):
    acc = 1
    for i in range(0, len(a)):
        acc *= a[i]
    return acc

#print(net.params)
weight_index = 0
for i_key,key_i in enumerate(network_keys):
    try:
        if 'data' == key_i or '_scale' in key_i:
            continue
        elif 'conv' in key_i:
            print(type(net.params[key_i][0].data), net.params[key_i][0].data.shape, type(net.params[key_i][0].data.shape))
	    net.params[key_i][0].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)]
            weight_index += get_weight_num(net.params[key_i][0].data.shape)
            if len(net.params[key_i]) > 1:
                print('has bias', key_i, len(net.params[key_i]), get_weight_num(net.params[key_i][1].data.shape))
                net.params[key_i][1].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][1].data.shape)]
                weight_index += get_weight_num(net.params[key_i][1].data.shape)
        elif 'bn' in key_i:
            print(type(net.params[key_i][0].data), net.params[key_i][0].data.shape, type(net.params[key_i][0].data.shape))
            #continue
            #key_mx = key_i + '_moving_mean'
            net.params[key_i][0].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)]
            net.params[key_i][2].data[...] = 1
            #print(weight_index, get_weight_num(net.params[key_i][0].data.shape),
            #      darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)][0:10])
            weight_index += get_weight_num(net.params[key_i][0].data.shape)

            #key_mx = key_i + '_moving_var'
            net.params[key_i][2].data[...] = 1
            net.params[key_i][1].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)]
            weight_index += get_weight_num(net.params[key_i][0].data.shape)

            #key_mx = key_i + '_gamma'
            net.params[key_i + '_scale'][0].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)]
            weight_index += get_weight_num(net.params[key_i][0].data.shape)

            net.params[key_i + '_scale'][1].data.flat = darknet_weight[weight_index : weight_index + get_weight_num(net.params[key_i][0].data.shape)]
            weight_index += get_weight_num(net.params[key_i][0].data.shape)
            
        else:
            sys.exit("Warning!    Unknown mxnet:{}".format(key_i))
        print("% 3d | %s initialized." %(i_key, key_i.ljust(40)))
    except KeyError as e:
        traceback.print_exc()
        exit()
net.save(args.cf_model)
print("\n- Finished.\n")

