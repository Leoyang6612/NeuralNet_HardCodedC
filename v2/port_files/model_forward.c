#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "layer_struct.h"
#include "model_forward.h"
#include "acti_forward.h"
#include "matrix_op.h"

// #define PRINT_LAYER_SHAPE
// #define PRINT_LAYER_OUTPUT

void normalize_forward(InputLayer *layer)
{

    //     | _ | _ |
    // len | _ | _ |
    //     | _ | _ |
    //       depth
    // example: len, depth = (3, 2) = (dim0, dim1)

    // MinMax-Scaler: transform input to (0,1)
    InputInfo info = layer->info;
    int input_length = info.dim0;
    int input_depth = info.dim1;

    float *input = layer->input;
    float *output = layer->output;

    float *min_arr = (float *)calloc(input_depth, sizeof(float));
    float *max_arr = (float *)calloc(input_depth, sizeof(float));

    // get min & max of depth(features)
    memcpy(min_arr, input, input_depth * sizeof(float));
    memcpy(max_arr, input, input_depth * sizeof(float));

#ifdef PRINT_LAYER_SHAPE
    printf("Now normalize...\n");
    printf("(%d, %d) => (%d, %d)\n", input_length, input_depth, input_length, input_depth);
#endif

    for (int i = 1; i < input_length; i++)
    {
        for (int j = 0; j < input_depth; j++)
        {
            float curr_val = input[i * input_depth + j];
            min_arr[j] = (min_arr[j] > curr_val) ? curr_val : min_arr[j];
            max_arr[j] = (max_arr[j] < curr_val) ? curr_val : max_arr[j];
        }
    }

    for (int i = 0; i < input_length; i++)
    {
        for (int j = 0; j < input_depth; j++)
        {
            output[i * input_depth + j] = (input[i * input_depth + j] - min_arr[j]) / (max_arr[j] - min_arr[j]);
        }
    }

    free(min_arr);
    free(max_arr);

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < input_length; i++)
    {
        for (int j = 0; j < input_depth; j++)
        {
            printf("%f ", output[i * input_depth + j]);
        }
        printf("\n");
    }
#endif
}

void conv1d_forward(Conv1dLayer *layer)
{
    // input (5, 3)
    // padding = 1   不需要真的padding，conv1d 運算的時候跳過即可
    // P  P  P
    // 01 02 03
    // 04 05 06
    // 07 08 09
    // 10 11 12
    // 13 14 15
    // P  P  P

    Conv1dInfo info = layer->info;
    int input_length = info.in_dim0;    // input_dim
    int input_features = info.in_dim1;  // input_features
    int output_features = info.filters; // output_features
    int kernel_size = info.kernel_size;
    int padding = info.padding;
    int stride = info.stride;
    float *weight = info.weight;
    float *bias = info.bias;
    ActivationType act = info.act;

    float *input = layer->input;
    float *output = layer->output; // has been padded. if padding != 0

    int output_length = (input_length - kernel_size + 2 * padding) / stride + 1;
#ifdef PRINT_LAYER_SHAPE
    printf("Now Conv1D...\n");
    printf("(%d, %d) => (%d, %d)\n", input_length, input_features, output_length, output_features);
#endif

    for (int f = 0; f < output_features; f++)
    {
        for (int i = 0; i < output_length; i++)
        {
            float conv_sum = 0.0f;
            for (int j = 0; j < kernel_size; j++)
            {

                // 不必幫input加入padding，已經考慮padding了
                // 比如 padding = 1
                // index, or you can say 'nth' timestep
                int index = i * stride + j - padding;
                if (index < 0 || index >= input_length)
                    continue;
                for (int k = 0; k < input_features; k++)
                {
                    conv_sum += input[index * input_features + k] * weight[j * (input_features * output_features) + k * output_features + f];
                    // conv_sum += input[index][k] * weight[j][k][f];
                }
            }
            output[i * output_features + f] = (conv_sum + bias[i]);
            // output[i][f] = (conv_sum + bias[i]);
        }
    }

    acti_forward(act, output, output_length * output_features);

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < output_length; i++)
    {
        for (int f = 0; f < output_features; f++)
        {
            printf("%.4f ", output[i * output_features + f]);
        }
        printf("\n");
    }
#endif
}

void dense_forward(DenseLayer *layer)
{
    DenseInfo info = layer->info;
    int input_length = info.in_dim0;
    int output_length = info.units;
    float *weight = info.weight;
    float *bias = info.bias;
    ActivationType act = info.act;

    float *input = layer->input;
    float *output = layer->output;
#ifdef PRINT_LAYER_SHAPE
    printf("Now Dense...\n");
    printf("(%d,) => (%d,)\n", input_length, output_length);
#endif

    // y = Wx
    vector_map(output, weight, input, input_length, output_length, BUFFER_STATE_OVERWRITE);
    // y = x + b
    vector_pointwise_add(output, output, bias, output_length, BUFFER_STATE_OVERWRITE);
    acti_forward(act, output, output_length);

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < output_length; i++)
    {
        printf("%.4f ", output[i]);
    }
    printf("\n");
#endif
}

void lstm_forward(LstmLayer *layer)
{
    LstmInfo info = layer->info;
    int time_step = info.in_dim0; // time_step
    int input_dim = info.in_dim1; // input_dim
    int units = info.units;       // output_len

    // input_weight (input_dim, 4 * units)
    // input gate
    float *W_i = info.weight_i;
    float *U_i = info.rcr_weight_i;
    float *b_i = info.bias_i;

    // forget gate
    float *W_f = info.weight_f;
    float *U_f = info.rcr_weight_f;
    float *b_f = info.bias_f;

    // new candidate cell state: Ct_bar
    float *W_c = info.weight_c;
    float *U_c = info.rcr_weight_c;
    float *b_c = info.bias_c;

    // output gate
    float *W_o = info.weight_o;
    float *U_o = info.rcr_weight_o;
    float *b_o = info.bias_o;

    bool return_seq = info.return_seq;

    float *input = layer->input;
    float *output = layer->output;

#ifdef PRINT_LAYER_SHAPE
    printf("Now Lstm...\n");
    if (return_seq)
    {
        printf("(%d, %d) => (%d, %d)\n", time_step, input_dim, time_step, units);
    }
    else
    {
        printf("(%d, %d) => (%d,)\n", time_step, input_dim, units);
    }

#endif

    // # input & output
    // current time input: Xt
    // current time output: ht
    // last time output: ht_1
    // current time hidden state: Ct
    // last time hidden state: Ct_1

    float *Xt = layer->Xt;
    float *ht = layer->ht;
    float *ht_1 = layer->ht_1;
    float *Ct = layer->Ct;
    float *Ct_1 = layer->Ct_1;

    // # buffers of 4 Gates:
    // input gate: it
    // forget gate: ft
    // new candidate cell state: Ct_bar
    // output gate: Ot
    float *it = layer->it;
    float *ft = layer->ft;
    float *Ct_bar = layer->Ct_bar;
    float *Ot = layer->Ot;

    for (int t = 0; t < time_step; t++)
    {
        // input gate (it)
        // it = sigmoid(Uiht−1 + Wixt + bi)
        memcpy(Xt, &input[t * input_dim], input_dim * sizeof(float));
        // Wi * xt
        vector_map(it, W_i, Xt, input_dim, units, BUFFER_STATE_OVERWRITE);
        // Ui * ht−1
        vector_map(it, U_i, ht_1, units, units, BUFFER_STATE_ACCUMULATE);
        vector_pointwise_add(it, it, b_i, units, BUFFER_STATE_OVERWRITE);
        acti_forward(ACTI_TYPE_HARDSIGMOID, it, units);

        // printf("it\n");
        // for (int i = 0; i < units; i++)
        // {
        //     printf("%.4f ", it[i]);
        // }
        // printf("\n");

        // forget gate (ft)
        // ft = sigmoid(Ufht−1 + Wfxt + bf)
        // Wf * xt
        vector_map(ft, W_f, Xt, input_dim, units, BUFFER_STATE_OVERWRITE);
        // Uf * ht−1
        vector_map(ft, U_f, ht_1, units, units, BUFFER_STATE_ACCUMULATE);
        vector_pointwise_add(ft, ft, b_f, units, BUFFER_STATE_OVERWRITE);
        acti_forward(ACTI_TYPE_HARDSIGMOID, ft, units);

        // printf("ft\n");
        // for (int i = 0; i < units; i++)
        // {
        //     printf("%.4f ", ft[i]);
        // }
        // printf("\n");

        // new candidate cell state
        // Ct_bar = tanh(Ucht−1 + Wcxt + bc)
        // Wc * xt
        vector_map(Ct_bar, W_c, Xt, input_dim, units, BUFFER_STATE_OVERWRITE);
        // Uc * ht−1
        vector_map(Ct_bar, U_c, ht_1, units, units, BUFFER_STATE_ACCUMULATE);
        vector_pointwise_add(Ct_bar, Ct_bar, b_c, units, BUFFER_STATE_OVERWRITE);
        acti_forward(ACTI_TYPE_TANH, Ct_bar, units);

        // printf("Ct_bar\n");
        // for (int i = 0; i < units; i++)
        // {
        //     printf("%.4f ", ft[i]);
        // }
        // printf("\n");

        // output gate (Ot)
        // ot = sigmoid(Uoht−1 + Woxt + bo)
        // Wo * xt
        vector_map(Ot, W_o, Xt, input_dim, units, BUFFER_STATE_OVERWRITE);
        // Uo * ht−1
        vector_map(Ot, U_o, ht_1, units, units, BUFFER_STATE_ACCUMULATE);
        vector_pointwise_add(Ot, Ot, b_o, units, BUFFER_STATE_OVERWRITE);
        acti_forward(ACTI_TYPE_HARDSIGMOID, Ot, units);
        // printf("Ot\n");
        // for (int i = 0; i < units; i++)
        // {
        //     printf("%.4f ", Ot[i]);
        // }
        // printf("\n");

        // ct = ft∗ct−1 + it∗ct_bar
        vector_pointwise_mul(Ct, ft, Ct_1, units, BUFFER_STATE_OVERWRITE);
        vector_pointwise_mul(Ct, it, Ct_bar, units, BUFFER_STATE_ACCUMULATE);
        // printf("Ct\n");
        // for (int i = 0; i < units; i++)
        // {
        //     printf("%.4f ", Ct[i]);
        // }
        // printf("\n");

        // will change value of Ct later
        memcpy(Ct_1, Ct, units * sizeof(float));

        // ht = ot∗tanh(ct)
        acti_forward(ACTI_TYPE_TANH, Ct, units);
        vector_pointwise_mul(ht, Ot, Ct, units, BUFFER_STATE_OVERWRITE);

        if (return_seq)
        {
            // copy ht to output
            for (int i = 0; i < units; i++)
            {
                output[t * units + i] = ht[i];
                // output[t][i] = ht[i];
            }
        }
        else if (t == time_step - 1)
        {
            // last timestep
            for (int i = 0; i < units; i++)
            {
                output[i] = ht[i];
            }
        }

        ht_1 = ht;
    }

#ifdef PRINT_LAYER_OUTPUT
    if (return_seq)
    {
        for (int t = 0; t < time_step; t++)
        {
            for (int i = 0; i < units; i++)
            {
                printf("%.4f ", output[t * units + i]);
            }
            printf("\n");
        }
    }
    else
    {
        for (int i = 0; i < units; i++)
        {
            printf("%.4f ", output[i]);
        }
        printf("\n");
    }
#endif
}

void avg_pool1d_forward(AvgPool1dLayer *layer)
{
    AvgPool1dInfo info = layer->info;
    int input_length = info.in_dim0; // input_len
    int input_features = info.in_dim1;
    int pool_size = info.pool_size;

    float *input = layer->input;
    float *output = layer->output;

    int output_length = input_length / pool_size;
    int output_features = input_features;
#ifdef PRINT_LAYER_SHAPE
    printf("Now Avgpool1D...\n");
    printf("(%d, %d) => (%d, %d)\n", input_length, input_features, output_length, output_features);
#endif

    for (int i = 0; i < output_length; i++)
    {
        for (int j = 0; j < input_features; j++)
        {

            float sum = 0.0f;
            // sum up all the nums in pool
            for (int k = 0; k < pool_size; k++)
            {
                // sum += input[pool_size * i + k][j];
                sum += input[(pool_size * i + k) * input_features + j];
            }

            // output[i][j] = sum / pool_size;
            output[i * input_features + j] = sum / pool_size;
        }
    }

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < output_length; i++)
    {
        for (int j = 0; j < output_features; j++)
        {
            printf("%.4f ", output[i * output_features + j]);
        }
        printf("\n");
    }
#endif
}

void max_pool1d_forward(MaxPool1dLayer *layer)
{
    // input: (16, 64)
    // pool_size = 2
    // output: (8, 64)

    MaxPool1dInfo info = layer->info;
    int input_length = info.in_dim0; // input_len
    int input_features = info.in_dim1;
    int pool_size = info.pool_size;

    float *input = layer->input;
    float *output = layer->output;

    int output_length = input_length / pool_size;
    int output_features = input_features;
#ifdef PRINT_LAYER_SHAPE
    printf("Now Maxpool 1D...\n");
    printf("(%d, %d) => (%d, %d)\n", input_length, input_features, output_length, output_features);
#endif

    for (int i = 0; i < output_length; i++)
    {
        for (int j = 0; j < input_features; j++)
        {
            // int max = input[pool_size * i][j];
            float max = input[pool_size * i * input_features + j];

            for (int k = 1; k < pool_size; k++)
            {
                // max vs input[pool_size * i + k][j]
                if (max < input[(pool_size * i + k) * input_features + j])
                {
                    max = input[(pool_size * i + k) * input_features + j];
                }
            }
            // output[i][j] = max;
            output[i * output_features + j] = max;
        }
    }

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < output_length; i++)
    {
        for (int j = 0; j < output_features; j++)
        {
            printf("%.4f ", output[i * output_features + j]);
        }
        printf("\n");
    }
#endif
}