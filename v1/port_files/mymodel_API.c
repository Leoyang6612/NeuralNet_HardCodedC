#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "layer_structure.h"

#define DEBUG_FLAG 0

void ReLU(float *input, int input_len)
{
    for (int i = 0; i < input_len; i++)
    {
        if (input[i] < 0)
        {
            input[i] = 0;
        }
    }
    return;
}

void Softmax(float *input, int input_len)
{
    float base = 0;
    for (int i = 0; i < input_len; i++)
    {
        base += exp(input[i]);
    }

    for (int j = 0; j < input_len; j++)
    {
        input[j] = exp(input[j]) / base;
    }
    return;
}

// LSTM default recurrent activation function in Keras
void HardSigmoid(float *input, uint8_t input_len)
{
    uint8_t len_i;
    for (len_i = 0; len_i < input_len; len_i++)
    {
        if (input[len_i] < -2.5)
        {
            input[len_i] = 0;
        }
        else if (input[len_i] > 2.5)
        {
            input[len_i] = 1;
        }
        else
        {
            input[len_i] = 0.2 * input[len_i] + 0.5;
        }
    }
    return;
}

void Sigmoid(float *input, uint8_t input_len)
{
    uint8_t len_i;
    for (len_i = 0; len_i < input_len; len_i++)
    {
        input[len_i] = 1 / (1 + exp(-1 * input[len_i]));
    }
    return;
}

void Tanh(float *input, uint8_t input_len)
{
    uint8_t len_i;
    for (len_i = 0; len_i < input_len; len_i++)
    {
        input[len_i] = (float)tanh((double)input[len_i]);
    }
    return;
}

void Activation_Func(ACTI_TYPE activation,
                     float *input,
                     int input_len)
{
    switch (activation)
    {
    case ACTI_TYPE_RELU:
        ReLU(input, input_len);
        break;
    case ACTI_TYPE_SOFTMAX:
        Softmax(input, input_len);
        break;
    case ACTI_TYPE_SIGMOID:
        Sigmoid(input, input_len);
        break;
    case ACTI_TYPE_HARDSIGMOID:
        HardSigmoid(input, input_len);
        break;
    case ACTI_TYPE_TANH:
        Tanh(input, input_len);
        break;
    default:
        break;
    }
#if DEBUG_FLAG != 0
    printf("Acti Function: %d\r\n", activation);
#endif
    return;
}

float *Arr_Copy(float *reference, uint8_t start, uint8_t len)
{
    float *toBeCopied = (float *)calloc(len, sizeof(float));
    for (uint8_t len_i = 0; len_i < len; len_i++)
    {
        toBeCopied[len_i] = reference[len_i + start];
    }
    return toBeCopied;
}

float *Arr_Plus(float *input1, float *input2, uint8_t len)
{
    float *output = (float *)calloc(len, sizeof(float));
    for (uint8_t len_i = 0; len_i < len; len_i++)
    {
        output[len_i] = input1[len_i] + input2[len_i];
    }
    free(input1);
    free(input2);
    return output;
}

float *Arr_Multi(float *input1, float *input2, uint8_t len)
{
    float *output = (float *)calloc(len, sizeof(float));
    for (uint8_t len_i = 0; len_i < len; len_i++)
    {
        output[len_i] = input1[len_i] * input2[len_i];
    }
    free(input1);
    free(input2);
    return output;
}

float Conv_ReLU(float *input,
                float *weight,
                float *bias,
                uint8_t kernel_size,
                uint8_t filter_depth,
                uint8_t filters_i,
                uint8_t len_index,
                uint8_t filters)
{
    uint8_t depth_i, kernel_i;
    float sum = 0;
    for (depth_i = 0; depth_i < filter_depth; depth_i++)
    {
        for (kernel_i = 0; kernel_i < kernel_size; kernel_i++)
        {
            sum += *(input + (kernel_i + len_index) * filter_depth + depth_i) * *(weight + kernel_i * filter_depth * filters + depth_i * filters + filters_i);
        }
    }
    sum += *(bias + filters_i);
    if (sum < 0)
        sum = 0;
    return sum;
}

float *Conv1D(const uint8_t filters,
              const uint8_t kernel_size,
              const uint8_t filter_depth,
              const uint8_t input_len,
              float *input,
              float *weight,
              float *bias)
{
    uint8_t filters_i, len_i, depth_i, kernel_i;
    uint8_t len = input_len - kernel_size + 1;

    float *output;
    output = (float *)calloc(len * filters, sizeof(float));
    if (output == NULL)
    {
        printf("Memory is not enough\n");
        while (1)
            ;
    }

    for (filters_i = 0; filters_i < filters; filters_i++)
    {
        for (len_i = 0; len_i < len; len_i++)
        {
            *(output + len_i * filters + filters_i) = Conv_ReLU(input, weight, bias, kernel_size, filter_depth, filters_i, len_i, filters);
        }
    }
    return output;
}

float *AveragePooling1D(float *input,
                        uint8_t pool_size,
                        uint8_t input_len,
                        uint8_t input_depth)
{
    uint8_t len_i, depth_i, pool_i;
    float result;
    float *output;
    uint8_t output_len;
    output_len = input_len / 2;
    output = (float *)calloc(output_len * input_depth, sizeof(float));
    if (output == NULL)
    {
        printf("Memory is not enough\n");
        while (1)
            ;
    }
    for (depth_i = 0; depth_i < input_depth; depth_i++)
    {
        for (len_i = 0; input_len - len_i > pool_size; len_i += pool_size)
        {
            result = 0;
            for (pool_i = 0; pool_i < pool_size; pool_i++)
            {
                if ((len_i + pool_i) >= input_len)
                    break;
                result += *(input + (len_i + pool_i) * input_depth + depth_i);
            }

            result /= (float)pool_i;
            *(output + (len_i / pool_size) * input_depth + depth_i) = result;
        }
    }
    return output;
}

float *Dense(float *input,
             float *weight,
             float *bias,
             uint8_t input_len,
             uint8_t output_len,
             ACTI_TYPE activation)
{
    uint8_t input_i, output_i;
    float sum;

    float *output;
    output = (float *)calloc(output_len, sizeof(float));
    if (output == NULL)
    {
        printf("Memory is not enough\n");
        while (1)
            ;
    }
    if (DEBUG_FLAG)
    {
        for (output_i = 0; output_i < output_len; output_i++)
        {
            for (input_i = 0; input_i < input_len; input_i++)
            {
                printf("%f ", weight[input_i * output_len + output_i]);
            }
        }
        printf("weight end\n\n");
    }

    for (output_i = 0; output_i < output_len; output_i++)
    {
        sum = 0;
        for (input_i = 0; input_i < input_len; input_i++)
        {
            sum += input[input_i] * weight[input_i * output_len + output_i];
        }
        sum += bias[output_i];
        output[output_i] = sum;
    }
    Activation_Func(activation, output, output_len);

#if DEBUG_FLAG != 0
    for (int i = 0; i < output_len; i++)
    {
        printf("\n%f ", output[i]);
    }
    printf("***output end***\n\n");
#endif

    return output;
}

float *LSTM_Conv(float *input,
                 float *weight,
                 uint8_t input_len,
                 uint8_t output_len,
                 int gate_index)
{
    float *output = (float *)calloc(output_len, sizeof(float));
    if (output == NULL)
    {
        printf("Memory is not enough\n");
        while (1)
            ;
    }
    // EXAMPLE:
    // Input shape & Weight shape
    // Xt (1,) (in a single time step)
    // in_weight (1, 256)
    // ht_1 (64,)
    // re_weight (64, 256)
    for (uint8_t output_i = 0; output_i < output_len; output_i++)
    {
        for (uint8_t input_i = 0; input_i < input_len; input_i++)
        {
            output[output_i] += input[input_i] * weight[input_i * output_len * 4 + output_i + gate_index];
        }
    }
    return output;
}

void LSTM_Bias(float *input,
               float *bias,
               uint8_t len,
               uint8_t gate_index)
{
    for (uint8_t len_i = 0; len_i < len; len_i++)
    {
        input[len_i] += bias[len_i + gate_index];
    }
    return;
}

float *Lstm(float *input,
            float *in_weight,
            float *re_weight,
            float *bias,
            uint8_t time_step,
            uint8_t input_len,
            uint8_t output_len,
            bool return_seq)
{
    // I / O:
    // current time input Xt
    // current time output ht
    // current time hidden state Ct
    // last time output ht_1
    // last time hidden state Ct_1
    float *Xt = NULL;
    float *ht = NULL;
    float *Ct = NULL;
    float *ht_1 = (float *)calloc(output_len, sizeof(float));
    float *Ct_1 = (float *)calloc(output_len, sizeof(float));

    // ht_all are ht at each time step
    float *ht_all = NULL;
    if (return_seq == true)
    {
        ht_all = (float *)calloc(time_step * output_len, sizeof(float));
    }

    // 4 Gates:
    //      input gate: it
    //      forget gate: ft
    //      new candidate cell state: Ct_bar
    //      output gate: Ot
    float *it = NULL;
    float *ft = NULL;
    float *Ct_bar = NULL;
    float *Ot = NULL;
    float *temp = NULL;

    for (uint8_t step = 0; step < time_step; step++)
    {
        if (DEBUG_FLAG)
        {
            printf("++++++++++++++%d iteration+++++++++++++++\n", step);
        }
        Xt = Arr_Copy(input, input_len * step, input_len);

        if (DEBUG_FLAG)
        {
            printf("Input Length: %d  Output Length: %d  TimeStep: %d\n", input_len, output_len, time_step);
        }

        // input gate (it)
        it = Arr_Plus(
            LSTM_Conv(Xt, in_weight, input_len, output_len, 0),
            LSTM_Conv(ht_1, re_weight, output_len, output_len, 0),
            output_len);
        LSTM_Bias(it, bias, output_len, 0);
        HardSigmoid(it, output_len);

        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len; i++)
            {
                printf("%f ", it[i]);
            }
            printf("\n***it end***\n");
        }

        // forget gate (ft)
        ft = Arr_Plus(
            LSTM_Conv(Xt, in_weight, input_len, output_len, output_len * 1),
            LSTM_Conv(ht_1, re_weight, output_len, output_len, output_len * 1),
            output_len * 1);
        LSTM_Bias(ft, bias, output_len, output_len * 1);
        HardSigmoid(ft, output_len);

        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len; i++)
            {
                printf("%f ", ft[i]);
            }
            printf("\n***ft end***\n");
        }

        // new candidate cell state (Ct_bar)
        Ct_bar = Arr_Plus(
            LSTM_Conv(Xt, in_weight, input_len, output_len, output_len * 2),
            LSTM_Conv(ht_1, re_weight, output_len, output_len, output_len * 2),
            output_len);
        LSTM_Bias(Ct_bar, bias, output_len, output_len * 2);
        Tanh(Ct_bar, output_len);

        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len; i++)
            {
                printf("%f ", Ct_bar[i]);
            }
            printf("\n***Ct_bar end***\n");
        }

        // output gate (Ot)
        Ot = Arr_Plus(
            LSTM_Conv(Xt, in_weight, input_len, output_len, output_len * 3),
            LSTM_Conv(ht_1, re_weight, output_len, output_len, output_len * 3),
            output_len);
        LSTM_Bias(Ot, bias, output_len, output_len * 3);
        HardSigmoid(Ot, output_len);

        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len; i++)
            {
                printf("%f ", Ot[i]);
            }
            printf("\n***Ot end***\n");
        }

        // last time to use ht_1 and Xt
        if (ht_1 != NULL)
            free(ht_1);
        else
            printf("ht_1 NULL\n\n");

        if (Xt != NULL)
            free(Xt);
        else
            printf("Xt NULL\n\n");

        Ct = Arr_Plus(
            // it, Ct_bar free here
            Arr_Multi(it, Ct_bar, output_len),
            // Ct_1, ft free here
            Arr_Multi(ft, Ct_1, output_len),
            output_len);

        temp = Arr_Copy(Ct, 0, output_len);
        Tanh(temp, output_len);
        ht = Arr_Multi(temp, Ot, output_len); // temp, Ot free here

        // copy ht to ht_all
        if (return_seq)
        {
            for (int i = 0; i < output_len; i++)
            {
                ht_all[i + output_len * step] = ht[i];
            }
        }

        Ct_1 = Ct;
        ht_1 = ht;

        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len; i++)
            {
                printf("%f ", ht[i]);
            }
            printf("\n***ht end***\n");
        }
    }
    if (Ct != NULL)
        free(Ct);
    else
        printf("Ct NULL\n\n");

    if (return_seq)
    {
        if (DEBUG_FLAG)
        {
            for (int i = 0; i < output_len * time_step; i++)
            {
                printf("%f ", ht_all[i]);
            }
            printf("\n***ht_all end***\n");
        }

        if (ht != NULL)
            free(ht);
        else
            printf("ht NULL\n\n");

        return ht_all;
    }

    return ht;
}

float *Standardize(float *input, uint8_t len, uint8_t depth)
{
    uint8_t len_i, depth_i;
    float *output = (float *)calloc(len * depth, sizeof(float));
    float mean, std_dev, sum_x, sum_x_square;
    for (depth_i = 0; depth_i < depth; depth_i++)
    {
        sum_x = 0;
        sum_x_square = 0;
        for (len_i = 0; len_i < len; len_i++)
        {
            sum_x += input[len_i * depth + depth_i];
            sum_x_square += input[len_i * depth + depth_i] * input[len_i * depth + depth_i];
        }
        mean = sum_x / (float)len;

        // sigma = sqrt((summation of x) / len - mean**2)
        std_dev = sqrt(sum_x_square / (float)len - mean * mean);
        printf("mean & standard deviation : %f  %f\n", mean, std_dev);

        for (len_i = 0; len_i < len; len_i++)
        {
            output[len_i * depth + depth_i] = (input[len_i * depth + depth_i] - mean) / std_dev;
        }
    }
    return output;
}

float *Normalize(float *input, uint8_t len, uint8_t depth)
{
    //     | _ | _ |
    // len | _ | _ |
    //     | _ | _ |
    //       depth
    // example: len, depth = (3, 2)

    // MinMax Scaler transform input to (0,1)
    float *mm_min = (float *)calloc(depth, sizeof(float));
    float *mm_max = (float *)calloc(depth, sizeof(float));
    float *mm_diff = (float *)calloc(depth, sizeof(float));
    memcpy(mm_min, input, depth * sizeof(float));
    memcpy(mm_max, input, depth * sizeof(float));

    // get min & max of depth(features)
    for (int i = 1; i < len; i++)
    {
        for (int j = 0; j < depth; j++)
        {
            int curr_input = input[i * depth + j];
            mm_min[j] = (mm_min[j] > curr_input) ? curr_input : mm_min[j];
            mm_max[j] = (mm_max[j] < curr_input) ? curr_input : mm_max[j];
        }
    }
#if DEBUG_FLAG != 0
    printf("(Normalize)min %f, max %f\r\n", mm_min[0], mm_max[0]);
#endif
    for (int i = 0; i < depth; i++)
    {
        mm_diff[i] = (mm_max[i] - mm_min[i]);
    }

    float *output = (float *)calloc(len * depth, sizeof(float));

    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < depth; j++)
        {
            output[i * depth + j] = (input[i * depth + j] - mm_min[j]) / mm_diff[j];
        }
    }
    free(mm_min);
    free(mm_max);
    free(mm_diff);

    return output;
}

void Next_Layer_Input(LAYER *p, float *input)
{
    switch (p->next_layer->type)
    {
    case LAYER_TYPE_OUTPUT:
        p->next_layer->output_layer->output = input;
        break;
    case LAYER_TYPE_INPUT:
        p->next_layer->conv1d_layer->input = input;
        break;
    case LAYER_TYPE_AVERAGE_POOL1D:
        p->next_layer->avg_pool1D_layer->input = input;
        break;
    case LAYER_TYPE_DENSE:
        p->next_layer->dense_layer->input = input;
        break;
    case LAYER_TYPE_LSTM:
        p->next_layer->lstm_layer->input = input;
        break;
    default:
        break;
    }
    return;
}

void Input_Layer(LAYER *p)
{
    INPUT_LAYER *layer;
    float *output;
    layer = p->input_layer;
    output = Normalize(layer->input, layer->len, layer->depth);
    // output = Arr_Copy(layer->input, 0, layer->len * layer->depth);

    Next_Layer_Input(p, output);
#if DEBUG_FLAG != 0
    printf("Input Layer Done\r\n");
#endif
    return;
}

void Conv1D_Layer(LAYER *p)
{
    CONV1D_LAYER *layer;
    float *output;
    uint8_t i;
    layer = p->conv1d_layer;
    output = Conv1D(layer->filters,
                    layer->kernel_size,
                    layer->filter_depth,
                    layer->input_len,
                    layer->input,
                    layer->weight,
                    layer->bias);
    Next_Layer_Input(p, output);
    free(layer->input);
    printf("Conv1D Layer \r\n");
    return;
}

void Average_Pooling_1D_Layer(LAYER *p)
{
    AVG_POOL1D_LAYER *layer;
    float *output;
    uint8_t i;
    layer = p->avg_pool1D_layer;
    output = AveragePooling1D(layer->input,
                              layer->pool_size,
                              layer->input_len,
                              layer->input_depth);
    Next_Layer_Input(p, output);
    free(layer->input);
#if DEBUG_FLAG != 0
    printf("Average Pooling 1D Layer Done\n");
#endif
    return;
}

void Dense_Layer(LAYER *p)
{
    DENSE_LAYER *layer;
    float *output;
    layer = p->dense_layer;
    output = Dense(layer->input,
                   layer->weight,
                   layer->bias,
                   layer->input_len,
                   layer->output_len,
                   layer->activation);

    Next_Layer_Input(p, output);

    if (layer->input != NULL)
        free(layer->input);
    else
        printf("Dense Layer Input NULL\r\n");

#if DEBUG_FLAG != 0
    printf("Dense Layer Done\r\n");
#endif
    return;
}

void LSTM_Layer(LAYER *p)
{
    LSTM_LAYER *layer;
    float *output;
    layer = p->lstm_layer;
    output = Lstm(layer->input,
                  layer->input_weight,
                  layer->recurrent_weight,
                  layer->bias,
                  layer->time_step,
                  layer->input_len,
                  layer->output_len,
                  layer->return_seq);

    Next_Layer_Input(p, output);

    if (layer->input != NULL)
        free(layer->input);
    else
        printf("LSTM Layer Input NULL\n\n");

#if DEBUG_FLAG != 0
    printf("LSTM Layer Done\r\n");
#endif
    return;
}
