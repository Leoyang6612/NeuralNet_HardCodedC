#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "layer_struct.h"
#include "model_forward.h"
#include "acti_forward.h"

// #define DEBUG_FLAG
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

#ifdef DEBUG_FLAG
    printf("Now normalize...\n");
    printf("(%d, %d) => (%d, %d)\n", input_length, input_depth, input_length, input_depth);
#endif

    for (int i = 1; i < input_length; i++)
    {
        for (int j = 0; j < input_depth; j++)
        {
            int curr_val = input[i * input_depth + j];
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

    // debug message
    // for (int i = 0; i < input_length; i++)
    // {
    //     for (int j = 0; j < input_depth; j++)
    //     {
    //         printf("%f ", output[i * input_depth + j]);
    //     }
    //     printf("\n");
    // }
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
#ifdef DEBUG_FLAG
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
            // output[i][f] += (conv_sum + bias[i]);
        }
    }

    acti_forward(act, output, output_length * output_features);

    // 輸出卷積後的結果
    // for (int i = 0; i < output_length; i++)
    // {
    //     for (int f = 0; f < output_features; f++)
    //     {
    //         printf("%.2f ", output[i * output_features + f]);
    //     }
    //     printf("\n");
    // }
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
#ifdef DEBUG_FLAG
    printf("Now Dense...\n");
    printf("(%d,) => (%d,)\n", input_length, output_length);
#endif

    for (int i = 0; i < output_length; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < input_length; j++)
        {
            sum += input[j] * weight[j * output_length + i];
            // sum += input[j] * weight[j][i];
        }
        output[i] = (sum + bias[i]);
    }


    acti_forward(act, output, output_length);

    // for (int i = 0; i < output_length; i++)
    // {
    //     printf("%.2f ", output[i]);
    // }
    // printf("\n");
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
#ifdef DEBUG_FLAG
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

    // for (int i = 0; i < output_length; i++)
    // {
    //     for (int j = 0; j < output_features; j++)
    //     {
    //         printf("%.2f ", output[i * output_features + j]);
    //     }
    //     printf("\n");
    // }
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
#ifdef DEBUG_FLAG
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

    // for (int i = 0; i < output_length; i++)
    // {
    //     for (int j = 0; j < output_features; j++)
    //     {
    //         printf("%.2f ", output[i * output_features + j]);
    //     }
    //     printf("\n");
    // }
}