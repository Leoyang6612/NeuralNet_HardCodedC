# %%
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import math
# %%


class H5FileInterpreter:
    def __init__(self, model_path):
        self.input_layer_not_create = True
        cwd = Path(__file__).parent
        self.model_path = cwd / model_path

        model = load_model(model_path, compile=False)
        self.layers = model.layers
        self.layer_count = 1
        self.last_layer_ouptut_buffer = ""

        Path(f"{cwd}/port_files").mkdir(parents=True, exist_ok=True)

        self.hfile = open(cwd / 'port_files/weight.h', 'w')
        self.cfile = open(cwd / 'port_files/model.c', 'w')

        self.__make_include_lib()
        self.__make_predict_func()

    def __nparray_to_Cstring(self, weight):
        char_to_replace = {'[': '{',
                           ']': '}', }
        weight_str = np.array2string(weight, precision=4, separator=', ')
        for key, value in char_to_replace.items():
            weight_str = weight_str.replace(key, value)
        weight_str += ";\n"

        return weight_str

    def traverse_layer(self):
        input_shape = self.layers[0].input_shape[1:]
        print("Input shape:", input_shape)
        self.__create_input_layer(input_shape)
        self.layer_count += 1
        print("="*40)

        for layer in self.layers:

            print(f"Layer: {layer.name} ({type(layer).__name__})")
            if type(layer).__name__ in ["Flatten", "Dropout"]:
                print("="*40)
                continue

            input_shape = layer.input_shape[1:]
            print('Input shape', input_shape)

            # expected output_shape
            exp_output_shape = layer.output_shape[1:]
            print("Output shape:", exp_output_shape)

            output_shape = self.__create_layer(
                layer, input_shape)

            assert (exp_output_shape == output_shape), "Output shape is wrong!"
            weights = layer.get_weights()
            if len(weights) > 0:
                print("Weights shape:", weights[0].shape, weights[1].shape)
            print("="*40)
            self.layer_count += 1

        print("Output shape:", output_shape)
        self.__create_output_layer(output_shape)
        self.layer_count += 1

        self.cfile.close()
        self.hfile.close()

    def __create_layer(self, layer, input_shape):
        # 'layer_name': 'conv1d_14';  'class_name': 'Conv1D'

        layer_name = layer.name
        class_name = type(layer).__name__

        if class_name == 'Conv1D':
            output_shape = self.__create_conv1D_layer(layer, input_shape)
        elif class_name == 'Dense':
            output_shape = self.__create_dense_layer(layer, input_shape)
        elif class_name == 'LSTM':
            output_shape = self.__create_lstm_layer(layer, input_shape)
        elif class_name == 'AveragePooling1D':
            output_shape = self.__create_avgpooling1D_layer(layer, input_shape)
        elif class_name == 'MaxPooling1D':
            output_shape = self.__create_maxpooling1D_layer(layer, input_shape)
        else:
            print(f"{class_name} not support yet!")
            output_shape = input_shape

        return output_shape

    def __create_input_layer(self, input_shape):
        input_len = input_shape[0]
        input_dim = input_shape[1] if len(input_shape) >= 2 else 1

        self.cfile.write(f'''
Layer *load_model()
{{
    Layer *currLayer, *headptr;
    headptr = (Layer *)malloc(sizeof(Layer));
    currLayer = headptr;

    // Layer {self.layer_count}: input_layer
    float *input = (float *)malloc(({input_len} * {input_dim}) * sizeof(float));
    currLayer->type = LAYER_TYPE_INPUT;
    currLayer->name = strdup("Input");
    currLayer->input_layer = (InputLayer *)malloc(sizeof(InputLayer));
    currLayer->input_layer->exec = normalize_forward;
    // currLayer->input_layer->exec = NULL;
    currLayer->input_layer->input = input;
    currLayer->input_layer->output = input;

    InputInfo *input_info = &(currLayer->input_layer->info);
    input_info->dim0 = {input_len};
    input_info->dim1 = {input_dim};
    input_info->normalize = true;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = "input"

    def __create_output_layer(self, input_shape):
        if len(input_shape) < 2:
            input_len = input_shape[0]
        else:
            input_len = input_shape[0] * input_shape[1]

        self.cfile.write(f'''
    // Layer {self.layer_count}: output_layer
    currLayer->type = LAYER_TYPE_OUTPUT;
    currLayer->name = strdup("Output");
    currLayer->output_layer = (OutputLayer *)malloc(sizeof(OutputLayer));
    currLayer->output_layer->output = {self.last_layer_ouptut_buffer};

    OutputInfo *output_info = &(currLayer->output_layer->info);
    output_info->dim0 = {input_len};
    currLayer->next = NULL;

    return headptr;
}}
    ''')

    def __create_conv1D_layer(self, layer, input_shape):
        # Ex:
        # input_shape (Timestep, Feature)/stride
        # output_shape ceil( Timestep)/Stride )                     if padding = 'same'
        # output_shape ceil((Timestep - KernelSize + 1)/Stride)     if padding = 'valid'(default)

        # input_shape (8, 64) (18 timestep, 64 in_feature)
        # kernel_size(3, )
        # padding 'same' = pad 1(=(kernel_size-1) / 2) zero at both left and right side
        # output_shape (8, 32) (8 timestep, 32 out_feature)         if padding = 'same'
        # output_shape (6, 32) (6 timestep, 32 out_feature)         if padding = 'valid'(default)

        layer_name = layer.name

        input_len = int(input_shape[0])
        input_depth = int(input_shape[1])

        weight_name = f"{layer_name}_w"
        bias_name = f"{layer_name}_b"

        filters = layer.filters
        kernel_size = layer.kernel_size[0]
        stride = layer.strides[0]

        if layer.padding == 'valid':
            output_shape = (
                math.ceil((input_len - kernel_size + 1)/stride), filters)
            padding = 0
        elif layer.padding == 'same':
            output_shape = (math.ceil(input_len/stride), filters)
            padding = (kernel_size - 1) // 2

        output_len = output_shape[0]
        output_dim = output_shape[1]

        info_name = f"{layer_name}_info"
        output_buffer = f"{layer_name}_output"
        acti_name = f"ACTI_TYPE_{layer.activation.__name__.upper()}"
        self.cfile.write(f'''
    // Layer {self.layer_count}: {layer_name}
    float *{output_buffer} = (float *)malloc(({output_len} * {output_dim}) * sizeof(float));
    currLayer->type = LAYER_TYPE_CONV1D;
    currLayer->name = strdup("Conv1D");
    currLayer->conv1d_layer = (Conv1dLayer *)malloc(sizeof(Conv1dLayer));
    currLayer->conv1d_layer->exec = conv1d_forward;
    currLayer->conv1d_layer->input = {self.last_layer_ouptut_buffer};
    currLayer->conv1d_layer->output = {output_buffer};

    Conv1dInfo *{info_name} = &(currLayer->conv1d_layer->info);
    {info_name}->in_dim0 = {input_len}; // input_len
    {info_name}->in_dim1 = {input_depth};  // depth
    {info_name}->filters = {filters}; // filters
    {info_name}->kernel_size = {kernel_size};
    {info_name}->padding = {padding};
    {info_name}->stride = {stride};
    {info_name}->weight = (float *){weight_name};
    {info_name}->bias = (float *){bias_name};
    {info_name}->act = {acti_name};
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = output_buffer

        weights = layer.get_weights()
        weight_kernel = weights[0]
        assert weight_kernel.shape == (
            kernel_size, input_depth, filters), "Create_Conv1D_Layer weight incorrect!"
        Cstring = self.__nparray_to_Cstring(weight_kernel)
        self.hfile.write(
            f'float {weight_name}[{kernel_size}][{input_depth}][{filters}] = ')
        self.hfile.write(Cstring)

        weight_bias = weights[1]
        assert weight_bias.shape[0] == filters, "Create_Conv1D_Layer bias incorrect!"
        Cstring = self.__nparray_to_Cstring(weight_bias)
        self.hfile.write(f'float {bias_name}[{filters}] = ')
        self.hfile.write(Cstring)

        return output_shape

    def __create_dense_layer(self, layer, input_shape):
        layer_name = layer.name

        weight_name = f"{layer_name}_w"
        bias_name = f"{layer_name}_b"

        if len(input_shape) < 2:
            input_len = input_shape[0]
        else:
            input_len = input_shape[0] * input_shape[1]

        output_len = layer.units

        info_name = f"{layer_name}_info"
        output_buffer = f"{layer_name}_output"
        acti_name = f"ACTI_TYPE_{layer.activation.__name__.upper()}"

        self.cfile.write(f'''
    // Layer {self.layer_count}: {layer_name}
    float *{output_buffer} = (float *)malloc(({output_len}) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = {self.last_layer_ouptut_buffer};
    currLayer->dense_layer->output = {output_buffer};

    DenseInfo *{info_name} = &(currLayer->dense_layer->info);
    {info_name}->in_dim0 = {input_len};      // input_len
    {info_name}->units = {output_len};       // output_units
    {info_name}->weight = (float *){weight_name};
    {info_name}->bias = (float *){bias_name};
    {info_name}->act = {acti_name};
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = output_buffer

        weights = layer.get_weights()
        weight_kernel = weights[0]

        assert weight_kernel.shape == (
            input_len, output_len), 'Create_Dense_Layer weight incorrect!'
        Cstring = self.__nparray_to_Cstring(weight_kernel)
        self.hfile.write(f'float {weight_name}[{input_len}][{output_len}] = ')
        self.hfile.write(Cstring)

        weight_bias = weights[1]
        assert weight_bias.shape[0] == output_len, 'Create_Dense_Layer bias incorrect!'
        Cstring = self.__nparray_to_Cstring(weight_bias)
        self.hfile.write(f'float {bias_name}[{output_len}] = ')
        self.hfile.write(Cstring)

        return (output_len,)

    def __create_lstm_layer(self, layer, input_shape):
        layer_name = layer.name

        weight_name = f"{layer_name}_w"
        recurr_weight_name = f"{layer_name}_u"
        bias_name = f"{layer_name}_b"

        time_step = int(input_shape[0])
        input_features = int(input_shape[1])

        return_seq = layer.return_sequences
        units = layer.units
        output_len = units

        info_name = f"{layer_name}_info"
        output_buffer = f"{layer_name}_output"

        self.cfile.write(f'''
    // Layer {self.layer_count}: {layer_name}
    float *{output_buffer} = (float *)malloc(({time_step if return_seq else 1} * {units}) * sizeof(float));

    currLayer->type = LAYER_TYPE_LSTM;
    currLayer->name = strdup("Lstm");
    currLayer->lstm_layer = (LstmLayer *)malloc(sizeof(LstmLayer));
    currLayer->lstm_layer->exec = lstm_forward;
    currLayer->lstm_layer->input = {self.last_layer_ouptut_buffer};
    currLayer->lstm_layer->output = {output_buffer};

    currLayer->lstm_layer->Xt = (float *)calloc({input_features}, sizeof(float));
    currLayer->lstm_layer->it = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->ht = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->ht_1 = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->ft = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->Ct = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->Ct_1 = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->Ct_bar = (float *)calloc({units}, sizeof(float));
    currLayer->lstm_layer->Ot = (float *)calloc({units}, sizeof(float));

    LstmInfo *{info_name} = &(currLayer->lstm_layer->info);
    {info_name}->in_dim0 = {time_step}; // time_step
    {info_name}->in_dim1 = {input_features};  // features
    {info_name}->units = {units};
    {info_name}->return_seq = {"true" if return_seq else "false"};
    // input gate
    {info_name}->weight_i = (float *){weight_name}_i;
    {info_name}->rcr_weight_i = (float *){recurr_weight_name}_i;
    {info_name}->bias_i = (float *){bias_name}_i;
    // forget gate
    {info_name}->weight_f = (float *){weight_name}_f;
    {info_name}->rcr_weight_f = (float *){recurr_weight_name}_f;
    {info_name}->bias_f = (float *){bias_name}_f;
    // new candidate cell state: Ct_bar
    {info_name}->weight_c = (float *){weight_name}_c;
    {info_name}->rcr_weight_c = (float *){recurr_weight_name}_c;
    {info_name}->bias_c = (float *){bias_name}_c;
    // output gate
    {info_name}->weight_o = (float *){weight_name}_o;
    {info_name}->rcr_weight_o = (float *){recurr_weight_name}_o;
    {info_name}->bias_o = (float *){bias_name}_o;

    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = output_buffer

        weights = layer.get_weights()

        # weight (for input)
        weight_kernel = weights[0]
        assert weight_kernel.shape == (
            input_features, output_len * 4), 'Create_Lstm_Layer weight incorrect!'
        Cstring = self.__nparray_to_Cstring(weight_kernel[:, :units])
        self.hfile.write(
            f'float {weight_name}_i[{input_features}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(weight_kernel[:, units: units * 2])
        self.hfile.write(
            f'float {weight_name}_f[{input_features}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            weight_kernel[:, units * 2: units * 3])
        self.hfile.write(
            f'float {weight_name}_c[{input_features}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            weight_kernel[:, units * 3:])
        self.hfile.write(
            f'float {weight_name}_o[{input_features}][{output_len}] = ')
        self.hfile.write(Cstring)

        # recurrent weight
        recurr_weight_kernel = weights[1]
        assert recurr_weight_kernel.shape == (
            output_len, output_len * 4), 'Create_Lstm_Layer recurrent weight incorrect!'
        Cstring = self.__nparray_to_Cstring(recurr_weight_kernel[:, :units])
        self.hfile.write(
            f'float {recurr_weight_name}_i[{output_len}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            recurr_weight_kernel[:, units: units * 2])
        self.hfile.write(
            f'float {recurr_weight_name}_f[{output_len}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            recurr_weight_kernel[:, units * 2: units * 3])
        self.hfile.write(
            f'float {recurr_weight_name}_c[{output_len}][{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            recurr_weight_kernel[:, units * 3:])
        self.hfile.write(
            f'float {recurr_weight_name}_o[{output_len}][{output_len}] = ')
        self.hfile.write(Cstring)

        # bias
        weight_bias = weights[2]
        assert weight_bias.shape == (
            output_len * 4,), 'Create_Lstm_Layer bias incorrect!'
        Cstring = self.__nparray_to_Cstring(weight_bias[:units])
        self.hfile.write(
            f'float {bias_name}_i[{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(weight_bias[units: units * 2])
        self.hfile.write(
            f'float {bias_name}_f[{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            weight_bias[units * 2: units * 3])
        self.hfile.write(
            f'float {bias_name}_c[{output_len}] = ')
        self.hfile.write(Cstring)

        Cstring = self.__nparray_to_Cstring(
            weight_bias[units * 3:])
        self.hfile.write(
            f'float {bias_name}_o[{output_len}] = ')
        self.hfile.write(Cstring)

        if return_seq:
            return (time_step, output_len)
        else:
            return (output_len,)

    def __create_avgpooling1D_layer(self, layer, input_shape):
        layer_name = layer.name

        input_len = int(input_shape[0])
        input_depth = int(input_shape[1])

        pool_size = layer.pool_size[0]
        output_len = input_len // pool_size
        output_depth = input_depth

        info_name = f"{layer_name}_info"
        output_buffer = f"{layer_name}_output"
        self.cfile.write(f'''
    float *{output_buffer} = (float *)malloc(({output_len} * {output_depth}) * sizeof(float));
    
    // Layer {self.layer_count}: {layer_name}
    currLayer->type = LAYER_TYPE_MAX_POOL1D;
    currLayer->name = strdup("AvgPool1D");
    currLayer->max_pool1d_layer = (AvgPool1dLayer *)malloc(sizeof(AvgPool1dLayer));
    currLayer->max_pool1d_layer->exec = avg_pool1d_forward;
    currLayer->max_pool1d_layer->input = {self.last_layer_ouptut_buffer};
    currLayer->max_pool1d_layer->output = {output_buffer};

    AvgPool1dInfo *{info_name} = &(currLayer->avg_pool1d_layer->info);
    {info_name}->in_dim0 = {input_len}; // input_len
    {info_name}->in_dim1 = {input_depth}; // depth
    {info_name}->pool_size = {pool_size};
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = output_buffer

        return (output_len, output_depth)

    def __create_maxpooling1D_layer(self, layer, input_shape):
        layer_name = layer.name

        input_len = int(input_shape[0])
        input_depth = int(input_shape[1])

        pool_size = layer.pool_size[0]
        output_len = input_len // pool_size
        output_depth = input_depth

        info_name = f"{layer_name}_info"
        output_buffer = f"{layer_name}_output"
        self.cfile.write(f'''
    float *{output_buffer} = (float *)malloc(({output_len} * {output_depth}) * sizeof(float));
    
    // Layer {self.layer_count}: {layer_name}
    currLayer->type = LAYER_TYPE_MAX_POOL1D;
    currLayer->name = strdup("MaxPool1D");
    currLayer->max_pool1d_layer = (MaxPool1dLayer *)malloc(sizeof(MaxPool1dLayer));
    currLayer->max_pool1d_layer->exec = max_pool1d_forward;
    currLayer->max_pool1d_layer->input = {self.last_layer_ouptut_buffer};
    currLayer->max_pool1d_layer->output = {output_buffer};

    MaxPool1dInfo *{info_name} = &(currLayer->max_pool1d_layer->info);
    {info_name}->in_dim0 = {input_len}; // input_len
    {info_name}->in_dim1 = {input_depth}; // depth
    {info_name}->pool_size = {pool_size};
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    ''')
        self.last_layer_ouptut_buffer = output_buffer

        return (output_len, output_depth)

    def __make_predict_func(self):
        self.cfile.write('''
unsigned int predict(Layer *headptr, float *input)
{
    Layer *currLayer = headptr;

    InputLayer *input_layer = currLayer->input_layer;
    size_t size = input_layer->info.dim0 * input_layer->info.dim1;
    memcpy(input_layer->input, input, size * sizeof(float));

    if (input_layer->exec)
    {
        input_layer->exec(input_layer);
    }
    currLayer = currLayer->next;

    while (currLayer && currLayer->type != LAYER_TYPE_OUTPUT)
    {
        switch (currLayer->type)
        {
        case LAYER_TYPE_CONV1D:
        {
            Conv1dLayer *conv1d_layer = currLayer->conv1d_layer;
            conv1d_layer->exec(conv1d_layer);
            break;
        }

        case LAYER_TYPE_DENSE:
        {
            DenseLayer *dense_layer = currLayer->dense_layer;
            dense_layer->exec(dense_layer);
            break;
        }

        case LAYER_TYPE_LSTM:
        {
            LstmLayer *lstm_layer = currLayer->lstm_layer;
            lstm_layer->exec(lstm_layer);
            break;
        }

        case LAYER_TYPE_AVERAGE_POOL1D:
        {
            AvgPool1dLayer *avg_pool1d_layer = currLayer->avg_pool1d_layer;
            avg_pool1d_layer->exec(avg_pool1d_layer);
            break;
        }

        case LAYER_TYPE_MAX_POOL1D:
        {
            MaxPool1dLayer *max_pool1d_layer = currLayer->max_pool1d_layer;
            max_pool1d_layer->exec(max_pool1d_layer);
            break;
        }

        default:
            break;
        }
        currLayer = currLayer->next;
    }

    OutputLayer *layer = currLayer->output_layer;
    float *output = layer->output;
    int units = layer->info.dim0;

    unsigned int argmax = 0;
    float max_confidence = output[0];

    for (int i = 0; i < units; i++)
    {
        printf("%.4f\\n", output[i]);
    }

    for (int i = 1; i < units; i++)
    {
        if (output[i] > max_confidence)
        {
            max_confidence = output[i];
            argmax = i;
        }
    }
    return argmax;
}
 ''')

    def __make_include_lib(self):
        self.cfile.write('''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer_struct.h"
#include "model.h"
#include "model_forward.h"
#include "weight.h"
''')


# %%
if __name__ == "__main__":
    np.set_printoptions(suppress=True, threshold=np.inf, precision=4)

    myInterpreter = H5FileInterpreter('model/REhead_tfks_dnn_fft_CT.h5')
    myInterpreter.traverse_layer()
# %%
