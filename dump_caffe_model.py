import sys, argparse
import find_mxnet, find_caffe
import mxnet as mx
import caffe
import traceback

from array import array

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
#parser.add_argument('--mx-model',        type=str, default='/var/darknet/insightface/models/model-r50-am-lfw/model')
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/48net.prototxt')
parser.add_argument('--cf-model',        type=str, default='model_caffe/det3_bgr.caffemodel')
args = parser.parse_args()

net = caffe.Net(args.cf_prototxt, args.cf_model, caffe.TRAIN)
network_keys = []
f = open(args.cf_prototxt, 'rU')
for line in f.readlines():
    if 'name: "' not in line:
        continue
    if 'mxnet-' in line or 'ONet' in line: continue
    if ('bn' in line and '_scale' in line) or ('sc' in line and '_scale' in line):
        continue
    if '_plus' in line or 'dropout' in line:
        continue
    line = line[line.find('name: ') + 7 : -2]
    #print(line)
    network_keys.append(line)
f.close()

print(network_keys, len(network_keys))
weight_file = open('model.cnn', 'wb')
weight_start = array('i', [0, 2, 0, 0, 0])
weight_start.tofile(weight_file)
for i_key,key_i in enumerate(network_keys):
    try:
        if 'data' == key_i:
            continue
        elif 'conv' in key_i:
            if '-' in key_i and key_i != 'conv6-3': continue
            float_array = array('f', net.params[key_i][0].data.flat)
            #print 'conv', float_array[0:5]
            float_array.tofile(weight_file)
            if len(net.params[key_i]) == 2:
                #print(i_key, key_i, 'has bias')
                float_array = array('f', net.params[key_i][1].data.flat)
                float_array.tofile(weight_file)
            else:
                print(i_key, key_i, 'error')
        elif 'prelu' in key_i:
            #if key_i == 'prelu5': break
            assert (len(net.params[key_i]) == 1)
            float_array = array('f', net.params[key_i][0].data.flat)
            print 'prelu', float_array[0:5], len(float_array)
            float_array.tofile(weight_file)
        else:
            #print("Warning!    Unknown layer:{}".format(key_i))
            pass
        print("% 3d | %s initialized." %(i_key, key_i.ljust(40)))
    except KeyError as e:
        traceback.print_exc()
        exit()
weight_file.close()

#net.save(args.cf_model)
print("\n- Finished.\n")

