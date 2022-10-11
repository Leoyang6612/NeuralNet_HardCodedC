# %%
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from pathlib import Path
# %%


def makeIncludeHeader(cfile):
    cfile.write('''
#include <stdio.h>
#include <stdlib.h>
#include "layer_structure.h"
#include "mymodel_API.h"
#include "create_layer_API.h"
#include "header.h"
''')


def makePredictFunc(cfile):
    cfile.write('''
float* predict(float* input, LAYER* head)
{
    LAYER *layer = head;
    layer->input_layer->input = input;
    //printf("======== Inference  Start ========\\r\\n");
    while (layer->type != LAYER_TYPE_OUTPUT)
    {
        layer->func(layer);
        layer = layer->next_layer;
    }

    int units = layer->output_layer->units;
    float *output = layer->output_layer->output;

	unsigned int ret = 0;
    float max_confidence = output[0];
    printf("%.6f\\r\\n", output[0]);

    for (int i = 1; i < units; i++)
    {
        if (output[i] > max_confidence){
            max_confidence = output[i];
            ret = i;
        }
        printf("%.6f\\r\\n", output[i]);
    }
    //printf("======== Inference Finish ========\\r\\n");
	free(output);
    return ret;
}
 ''')


def Create_Input_Layer(layer, cfile):
    input_size = layer['config']['batch_input_shape']
    # EXAMPLE:
    # input size (10, 40, 1)
    # 10 batches with input shape=(40,1)
    length = input_size[1]
    depth = 1 if len(input_size) <= 2 else input_size[2]

    cfile.write(f'''
LAYER* load_model()
{{
    LAYER *layer, *layer_head;
    layer = (LAYER *)malloc(sizeof(LAYER));
    layer_head = layer;
    Create_Input_Layer(layer, {length}, {depth});
    layer = layer->next_layer;
''')
    return length, depth


def Create_Conv1D_Layer(layer_info, cfile, hfile, weight_key, model_weight, input_shape):
    filters = layer_info['filters']
    kernel_size = layer_info['kernel_size'][0]
    filter_depth = int(input_shape[1])
    input_len = int(input_shape[0])
    weight = layer_info['name'] + 'w'
    bias = layer_info['name'] + 'b'
    cfile.write('\tCreate_Conv1D_Layer(layer,')
    cfile.write(str(filters) + ',')
    cfile.write(str(kernel_size) + ',')
    cfile.write(str(filter_depth) + ',')
    cfile.write(str(input_len) + ',(float*)')
    cfile.write(weight + ',' + bias + ');\n\n')

    weight_arr = model_weight[weight_key]
    weight_arr = weight_arr[weight_key]
    weight_kernel = weight_arr['kernel:0']
    kernel_shape = weight_kernel.shape
    if kernel_shape[0] == kernel_size and kernel_shape[1] == filter_depth and kernel_shape[2] == filters:
        print('weight shape is right!!!')
    else:
        print(weight + ' shape is wrong ... QAQ')
    weight_bias = weight_arr['bias:0']
    if weight_bias.shape[0] == filters:
        print('bias shape is right!!!')
    else:
        print(bias + ' shape is wrong... QAQ')
    fcomma = False
    dcomma = False
    kcomma = False
    hfile.write('float ' + weight + '[' + str(kernel_size) + ']')
    hfile.write('[' + str(filter_depth) + '][' + str(filters) + '] = {')
    for ker in weight_kernel:
        if kcomma:
            hfile.write(',')
        else:
            kcomma = True
        hfile.write('{')
        dcomma = False

        for dep in ker:
            if dcomma:
                hfile.write(',')
            else:
                dcomma = True
            hfile.write('{')
            fcomma = False

            for filt in dep:
                if fcomma:
                    hfile.write(',')
                else:
                    fcomma = True
                hfile.write(str(filt))
            hfile.write('}\n')
        hfile.write('}\n')
    hfile.write('};\n\n\n')
    hfile.write('float ' + bias + '[' + str(filters) + '] = {')
    fcomma = False
    for filt in weight_bias:
        if fcomma:
            hfile.write(',')
        else:
            fcomma = True
        hfile.write(str(filt))
    hfile.write('};\n\n\n')
    return (input_len - kernel_size + 1), filters


def Create_Average_Pooling_1D_Layer(layer_info, cfile, input_shape):
    pool_size = layer_info['pool_size'][0]
    input_len = input_shape[0]
    input_depth = input_shape[1]
    cfile.write('\tCreate_Average_Pooling_1D_Layer(layer,')
    cfile.write(str(pool_size) + ',')
    cfile.write(str(input_len) + ',')
    cfile.write(str(input_depth) + ');\n\n')
    if (input_len / 2) == 0:
        input_len = input_len / 2
    else:
        input_len = (input_len - 1) / 2
    return input_len, input_depth


def Create_Flatten_Layer(layer, cfile, input_shape):
    if len(input_shape) >= 2:
        input_len = int(input_shape[0])
        input_depth = int(input_shape[1])

        cfile.write(f'''
    Create_Flatten_Layer(layer, {input_len}, {input_depth});
    ''')
        return (input_len * input_depth,)
    else:
        return input_shape


def Create_Dense_Layer(layer_info, cfile, hfile, weight_key, model_weight, input_shape):
    weight = layer_info['name'] + 'w'
    bias = layer_info['name'] + 'b'
    input_len = input_shape[0]
    output_len = layer_info['units']
    acti_name = layer_info['name'] + 'ac'
    acti_type = f"ACTI_TYPE_{layer_info['activation']}".upper()

    cfile.write(f'''
    Create_Dense_Layer(layer, (float*){weight}, {bias}, {input_len}, {output_len}, {acti_name});
    layer = layer->next_layer;
    ''')
    hfile.write(f'''
ACTI_TYPE {acti_name} = {acti_type};
''')

    weight_arr = model_weight[weight_key]
    weight_arr = weight_arr[weight_key]

    weight_kernel = np.array(weight_arr['kernel:0'])
    kernel_shape = weight_kernel.shape
    if kernel_shape[0] == input_len and kernel_shape[1] == output_len:
        print('weight shape is right!!!')
    else:
        print(weight + ' shape is wrong ... QAQ')

    hfile.write(f'float {weight}[{input_len}][{output_len}] = ')
    char_to_replace = {'[': '{',
                       ']': '}', }
    weight_str = np.array2string(weight_kernel, precision=4, separator=', ')
    for key, value in char_to_replace.items():
        weight_str = weight_str.replace(key, value)
    hfile.write(weight_str + ";\n")

    weight_bias = np.array(weight_arr['bias:0'])
    if weight_bias.shape[0] == output_len:
        print('bias shape is right!!!')
    else:
        print(bias + ' shape is wrong... QAQ')

    hfile.write(f'float {bias}[{output_len}] = ')
    weight_str = np.array2string(weight_bias, precision=4, separator=', ')
    for key, value in char_to_replace.items():
        weight_str = weight_str.replace(key, value)
    hfile.write(weight_str + ";\n")

    return (output_len,)


def Create_LSTM_Layer(layer_info, cfile, hfile, weight_key, model_weight, input_shape):
    input_weight = layer_info['name'] + 'inw'
    recurrent_weight = layer_info['name'] + 'rew'
    bias = layer_info['name'] + 'b'

    time_step, input_len = input_shape
    output_len = layer_info['units']

    if layer_info['return_sequences']:
        cfile.write(
            f'\tCreate_LSTM_Layer(layer, (float*){input_weight}, (float*){recurrent_weight}, {bias}, {time_step}, {input_len}, {output_len}, true);\n\n')
    else:
        cfile.write(
            f'\tCreate_LSTM_Layer(layer, (float*){input_weight}, (float*){recurrent_weight}, {bias}, {time_step}, {input_len}, {output_len}, false);\n\n')

    weight_arr = model_weight[weight_key]
    weight_arr = weight_arr[weight_key]
    weight_kernel = weight_arr['kernel:0']

    fcomma = False
    dcomma = False
    # double curly brace is used to escape the curly brace {{
    hfile.write(
        f'float {input_weight}[{weight_kernel.shape[0]}][{weight_kernel.shape[1]}] = {{')

    for input_node in weight_kernel:
        if dcomma:
            hfile.write(',')
        else:
            dcomma = True
        hfile.write('{')
        fcomma = False

        for output_node in input_node:
            if fcomma:
                hfile.write(',')
            else:
                fcomma = True
            hfile.write(str(output_node))
        hfile.write('}\n')
    hfile.write('};\n\n\n')

    re_weight_kernel = weight_arr['recurrent_kernel:0']

    fcomma = False
    dcomma = False
    hfile.write(
        f'float {recurrent_weight}[{re_weight_kernel.shape[0]}][{re_weight_kernel.shape[1]}] = {{')

    for input_node in re_weight_kernel:
        if dcomma:
            hfile.write(',')
        else:
            dcomma = True
        hfile.write('{')
        fcomma = False

        for output_node in input_node:
            if fcomma:
                hfile.write(',')
            else:
                fcomma = True
            hfile.write(str(output_node))
        hfile.write('}\n')
    hfile.write('};\n\n\n')

    weight_bias = weight_arr['bias:0']
    hfile.write(
        f'float {bias}[{weight_bias.shape[0]}] = {{')
    fcomma = False
    for output_b in weight_bias:
        if fcomma:
            hfile.write(',')
        else:
            fcomma = True
        hfile.write(str(output_b))
    hfile.write('};\n\n\n')

    if layer_info['return_sequences']:
        return (time_step, output_len)
    else:
        return (output_len,)


def Create_Layer(class_name, weight_key, model_weight, layer, cfile, hfile, input_shape):
    if layer['class_name'] == 'TimeDistributed':
        layer = layer['config']['layer']['config']
    else:
        layer = layer['config']
    if class_name == 'LSTM':
        input_shape = Create_LSTM_Layer(
            layer, cfile, hfile, weight_key, model_weight, input_shape)
        cfile.write('layer = layer->next_layer;\n')
    elif class_name == 'Conv1D':
        input_shape = Create_Conv1D_Layer(
            layer, cfile, hfile, weight_key, model_weight, input_shape)
        cfile.write('layer = layer->next_layer;\n')
    elif class_name == 'AveragePooling1D':
        input_shape = Create_Average_Pooling_1D_Layer(
            layer, cfile, input_shape)
        cfile.write('layer = layer->next_layer;\n')
    elif class_name == 'Flatten':
        input_shape = Create_Flatten_Layer(layer, cfile, input_shape)
    elif class_name == 'Dense':
        input_shape = Create_Dense_Layer(
            layer, cfile, hfile, weight_key, model_weight, input_shape)
    return input_shape


def Find_Layer_Type(layer):
    class_name = layer['class_name']
    if class_name == 'TimeDistributed':
        return layer['config']['layer']['class_name']
    else:
        return class_name


# %%
if __name__ == '__main__':
    input_layer_not_create = True
    cwd = Path(__file__).parent
    model_path = cwd / 'weight/head_tfks_dnn.h5'
    model = load_model(model_path)
    model_attrs = model.get_config()

    h5file = h5py.File(model_path, 'r')
    model_weight = h5file['model_weights']
    Path(f"{cwd}/port_files").mkdir(parents=True, exist_ok=True)
    hfile = open(cwd / 'port_files/header.h', 'w')
    cfile = open(cwd / 'port_files/model.c', 'w')
    makeIncludeHeader(cfile)
    input_shape = (0, 0)
    np.set_printoptions(suppress=True)

    for layer in model_attrs['layers']:
        print(layer, end='\n')
        weight_key = layer['config']['name']
        class_name = Find_Layer_Type(layer)

        if input_layer_not_create:
            input_shape = Create_Input_Layer(layer, cfile)
            input_layer_not_create = False
        if class_name == 'InputLayer':
            pass
        else:
            print('Input shape', input_shape)
            input_shape = Create_Layer(
                class_name, weight_key, model_weight, layer, cfile, hfile, input_shape)

    cfile.write(f'''
    Create_Output_Layer(layer, {input_shape[0]});
    return layer_head;
}}
    ''')

    makePredictFunc(cfile)
    hfile.close()
    cfile.close()

# %%
