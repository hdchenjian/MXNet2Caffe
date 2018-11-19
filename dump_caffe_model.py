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

net = caffe.Net(args.cf_prototxt, caffe.TRAIN)
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
            float_array = array('f', net.params[key_i][0].data.flat)
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
            if len(net.params[key_i]) >= 2:
                #print(i_key, key_i, 'has bias')
                float_array = array('f', net.params[key_i][1].data.flat)
                float_array.tofile(weight_file)
        elif 'bn' in key_i or 'fc' in key_i or 'sc' in key_i:
            print("warnig")
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
        elif 'prelu' in key_i:
            assert (len(net.params[key_i]) == 1)
            float_array = array('f', net.params[key_i][0].data.flat)
            float_array.tofile(weight_file)
        else:
            #sys.exit("Warning!    Unknown layer:{}".format(key_i))
            pass
        print("% 3d | %s initialized." %(i_key, key_i.ljust(40)))
    except KeyError as e:
        traceback.print_exc()
        exit()
weight_file.close()

#net.save(args.cf_model)
print("\n- Finished.\n")

