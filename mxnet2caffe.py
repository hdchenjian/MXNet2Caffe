import sys, argparse
import find_mxnet, find_caffe
import mxnet as mx
import caffe
import traceback

from array import array

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',        type=str, default='/var/darknet/insightface/models/model-r50-am-lfw/model')
parser.add_argument('--mx-epoch',        type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/face/facega2.prototxt')
parser.add_argument('--cf-model',        type=str, default='model_caffe/face/facega2.caffemodel')
args = parser.parse_args()

# ------------------------------------------
# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
net = caffe.Net(args.cf_prototxt, caffe.TRAIN)
network_keys = []
f = open(args.cf_prototxt, 'rU')
for line in f.readlines():
    if 'name: "' not in line:
        continue
    if 'mxnet-' in line: continue
    if ('bn' in line and '_scale' in line) or ('sc' in line and '_scale' in line):
        continue
    if '_plus' in line or 'dropout' in line:
        continue
    line = line[line.find('name: ') + 7 : -2]
    #print(line)
    network_keys.append(line)
f.close()

#print(network_keys)

weight_file = open('model.cnn', 'wb')
weight_start = array('i', [0, 2, 0, 0, 0])
weight_start.tofile(weight_file)
for i_key,key_i in enumerate(network_keys):
    try:
        if 'data' == key_i:
            continue
        elif 'conv' in key_i or 'pre_fc' in key_i:
            key_mx = key_i + '_weight'
	    net.params[key_i][0].data.flat = arg_params[key_mx].asnumpy().flat
            float_array = array('f', arg_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)
            '''
            print(type(net.params[key_i][0].data), len(arg_params[key_mx].asnumpy().flat), arg_params[key_mx].asnumpy().shape)
            print(len(float_array))
            print(float_array[0:10])
            print(float_array[-10:])
            count = 0
            for tmp in arg_params[key_mx].asnumpy().flat:
                print(count, tmp)
                count += 1
            '''
            key_mx = key_i + '_bias'
            if key_mx in arg_params:
                print(i_key, key_i, 'has bias')
                net.params[key_i][1].data.flat = arg_params[key_mx].asnumpy().flat
                float_array = array('f', arg_params[key_mx].asnumpy().flat)
                float_array.tofile(weight_file)
        elif 'bn' in key_i or 'fc' in key_i or 'sc' in key_i:
            key_mx = key_i + '_moving_mean'
            net.params[key_i][0].data.flat = aux_params[key_mx].asnumpy().flat
            net.params[key_i][2].data[...] = 1
            float_array = array('f', aux_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)

            key_mx = key_i + '_moving_var'
            net.params[key_i][2].data[...] = 1
            net.params[key_i][1].data.flat = aux_params[key_mx].asnumpy().flat
            float_array = array('f', aux_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)

            key_mx = key_i + '_gamma'
            net.params[key_i + '_scale'][0].data.flat = arg_params[key_mx].asnumpy().flat 
            float_array = array('f', arg_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)

            key_mx = key_i + '_beta'
            net.params[key_i + '_scale'][1].data.flat = arg_params[key_mx].asnumpy().flat 
            float_array = array('f', arg_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)
        elif 'relu' in key_i:
            assert (len(net.params[key_i]) == 1)
            key_mx = key_i + '_gamma'
            net.params[key_i][0].data.flat = arg_params[key_mx].asnumpy().flat
            float_array = array('f', arg_params[key_mx].asnumpy().flat)
            float_array.tofile(weight_file)
        else:
            sys.exit("Warning!    Unknown mxnet:{}".format(key_i))
        print("% 3d | %s initialized." %(i_key, key_i.ljust(40)))
    except KeyError as e:
        traceback.print_exc()
        exit()
weight_file.close()
'''
net_1 = caffe.Net(args.cf_prototxt, caffe.TRAIN)
all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()
for i_key,key_i in enumerate(all_keys):
    try:
        if 'data' is key_i:
            pass
        elif '_weight' in key_i:
            key_caffe = key_i.replace('_weight','')
            net_1.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            #print(net_1.params[key_caffe][0].data == net.params[key_caffe][0].data)
        elif '_bias' in key_i:
            key_caffe = key_i.replace('_bias','')
            net_1.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
            #print(net_1.params[key_caffe][1].data == net.params[key_caffe][1].data)
        elif '_gamma' in key_i and 'relu' not in key_i:
            key_caffe = key_i.replace('_gamma','_scale')
            net_1.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            #print(net_1.params[key_caffe][0].data == net.params[key_caffe][0].data)
        # TODO: support prelu
        elif '_gamma' in key_i and 'relu' in key_i:     # for prelu
            key_caffe = key_i.replace('_gamma','')
            assert (len(net_1.params[key_caffe]) == 1)
            net_1.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            #print(net_1.params[key_caffe][0].data == net.params[key_caffe][0].data)
        elif '_beta' in key_i:
            key_caffe = key_i.replace('_beta','_scale')
            net_1.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
            #print(net_1.params[key_caffe][1].data == net.params[key_caffe][1].data)
        elif '_moving_mean' in key_i:
            key_caffe = key_i.replace('_moving_mean','')
            net_1.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
            net_1.params[key_caffe][2].data[...] = 1
            #print(net_1.params[key_caffe][0].data == net.params[key_caffe][0].data)
        elif '_moving_var' in key_i:
            key_caffe = key_i.replace('_moving_var','')
            net_1.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
            net_1.params[key_caffe][2].data[...] = 1
            #print(net_1.params[key_caffe][1].data == net.params[key_caffe][1].data)
        else:
            sys.exit("Warning!    Unknown mxnet:{}".format(key_i))
        print("% 3d | %s -> %s, initialized." %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    except KeyError:
        print("\nError!  key error mxnet:{}".format(key_i))
'''
#net.save(args.cf_model)
print("\n- Finished.\n")

