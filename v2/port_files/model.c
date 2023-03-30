
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer_struct.h"
#include "model.h"
#include "model_forward.h"
#include "weight.h"

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
        printf("%.4f\n", output[i]);
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
 
Layer *load_model()
{
    Layer *currLayer, *headptr;
    headptr = (Layer *)malloc(sizeof(Layer));
    currLayer = headptr;

    // Layer 1: input_layer
    float *input = (float *)malloc((20 * 3) * sizeof(float));
    currLayer->type = LAYER_TYPE_INPUT;
    currLayer->name = strdup("Input");
    currLayer->input_layer = (InputLayer *)malloc(sizeof(InputLayer));
    currLayer->input_layer->exec = normalize_forward;
    // currLayer->input_layer->exec = NULL;
    currLayer->input_layer->input = input;
    currLayer->input_layer->output = input;

    InputInfo *input_info = &(currLayer->input_layer->info);
    input_info->dim0 = 20;
    input_info->dim1 = 3;
    input_info->normalize = true;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 2: conv1d_3
    float *conv1d_3_output = (float *)malloc((18 * 32) * sizeof(float));
    currLayer->type = LAYER_TYPE_CONV1D;
    currLayer->name = strdup("Conv1D");
    currLayer->conv1d_layer = (Conv1dLayer *)malloc(sizeof(Conv1dLayer));
    currLayer->conv1d_layer->exec = conv1d_forward;
    currLayer->conv1d_layer->input = input;
    currLayer->conv1d_layer->output = conv1d_3_output;

    Conv1dInfo *conv1d_3_info = &(currLayer->conv1d_layer->info);
    conv1d_3_info->in_dim0 = 20; // input_len
    conv1d_3_info->in_dim1 = 3;  // depth
    conv1d_3_info->filters = 32; // filters
    conv1d_3_info->kernel_size = 3;
    conv1d_3_info->padding = 0;
    conv1d_3_info->stride = 1;
    conv1d_3_info->weight = (float *)conv1d_3_w;
    conv1d_3_info->bias = (float *)conv1d_3_b;
    conv1d_3_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 3: conv1d_4
    float *conv1d_4_output = (float *)malloc((16 * 64) * sizeof(float));
    currLayer->type = LAYER_TYPE_CONV1D;
    currLayer->name = strdup("Conv1D");
    currLayer->conv1d_layer = (Conv1dLayer *)malloc(sizeof(Conv1dLayer));
    currLayer->conv1d_layer->exec = conv1d_forward;
    currLayer->conv1d_layer->input = conv1d_3_output;
    currLayer->conv1d_layer->output = conv1d_4_output;

    Conv1dInfo *conv1d_4_info = &(currLayer->conv1d_layer->info);
    conv1d_4_info->in_dim0 = 18; // input_len
    conv1d_4_info->in_dim1 = 32;  // depth
    conv1d_4_info->filters = 64; // filters
    conv1d_4_info->kernel_size = 3;
    conv1d_4_info->padding = 0;
    conv1d_4_info->stride = 1;
    conv1d_4_info->weight = (float *)conv1d_4_w;
    conv1d_4_info->bias = (float *)conv1d_4_b;
    conv1d_4_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    float *max_pooling1d_1_output = (float *)malloc((8 * 64) * sizeof(float));
    
    // Layer 4: max_pooling1d_1
    currLayer->type = LAYER_TYPE_MAX_POOL1D;
    currLayer->name = strdup("MaxPool1D");
    currLayer->max_pool1d_layer = (MaxPool1dLayer *)malloc(sizeof(MaxPool1dLayer));
    currLayer->max_pool1d_layer->exec = max_pool1d_forward;
    currLayer->max_pool1d_layer->input = conv1d_4_output;
    currLayer->max_pool1d_layer->output = max_pooling1d_1_output;

    MaxPool1dInfo *max_pooling1d_1_info = &(currLayer->max_pool1d_layer->info);
    max_pooling1d_1_info->in_dim0 = 16; // input_len
    max_pooling1d_1_info->in_dim1 = 64; // depth
    max_pooling1d_1_info->pool_size = 2;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 5: conv1d_5
    float *conv1d_5_output = (float *)malloc((8 * 32) * sizeof(float));
    currLayer->type = LAYER_TYPE_CONV1D;
    currLayer->name = strdup("Conv1D");
    currLayer->conv1d_layer = (Conv1dLayer *)malloc(sizeof(Conv1dLayer));
    currLayer->conv1d_layer->exec = conv1d_forward;
    currLayer->conv1d_layer->input = max_pooling1d_1_output;
    currLayer->conv1d_layer->output = conv1d_5_output;

    Conv1dInfo *conv1d_5_info = &(currLayer->conv1d_layer->info);
    conv1d_5_info->in_dim0 = 8; // input_len
    conv1d_5_info->in_dim1 = 64;  // depth
    conv1d_5_info->filters = 32; // filters
    conv1d_5_info->kernel_size = 3;
    conv1d_5_info->padding = 1;
    conv1d_5_info->stride = 1;
    conv1d_5_info->weight = (float *)conv1d_5_w;
    conv1d_5_info->bias = (float *)conv1d_5_b;
    conv1d_5_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 6: dense_2
    float *dense_2_output = (float *)malloc((32) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = conv1d_5_output;
    currLayer->dense_layer->output = dense_2_output;

    DenseInfo *dense_2_info = &(currLayer->dense_layer->info);
    dense_2_info->in_dim0 = 256;      // input_len
    dense_2_info->units = 32;       // output_units
    dense_2_info->weight = (float *)dense_2_w;
    dense_2_info->bias = (float *)dense_2_b;
    dense_2_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 7: dense_3
    float *dense_3_output = (float *)malloc((2) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = dense_2_output;
    currLayer->dense_layer->output = dense_3_output;

    DenseInfo *dense_3_info = &(currLayer->dense_layer->info);
    dense_3_info->in_dim0 = 32;      // input_len
    dense_3_info->units = 2;       // output_units
    dense_3_info->weight = (float *)dense_3_w;
    dense_3_info->bias = (float *)dense_3_b;
    dense_3_info->act = ACTI_TYPE_SOFTMAX;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 8: output_layer
    currLayer->type = LAYER_TYPE_OUTPUT;
    currLayer->name = strdup("Output");
    currLayer->output_layer = (OutputLayer *)malloc(sizeof(OutputLayer));
    currLayer->output_layer->output = dense_3_output;

    OutputInfo *output_info = &(currLayer->output_layer->info);
    output_info->dim0 = 2;
    currLayer->next = NULL;

    return headptr;
}
    