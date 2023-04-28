
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
    float *input = (float *)malloc((5 * 1) * sizeof(float));
    currLayer->type = LAYER_TYPE_INPUT;
    currLayer->name = strdup("Input");
    currLayer->input_layer = (InputLayer *)malloc(sizeof(InputLayer));
    currLayer->input_layer->exec = normalize_forward;
    // currLayer->input_layer->exec = NULL;
    currLayer->input_layer->input = input;
    currLayer->input_layer->output = input;

    InputInfo *input_info = &(currLayer->input_layer->info);
    input_info->dim0 = 5;
    input_info->dim1 = 1;
    input_info->normalize = true;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    // Layer 2: dense
    float *dense_output = (float *)malloc((64) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = input;
    currLayer->dense_layer->output = dense_output;

    DenseInfo *dense_info = &(currLayer->dense_layer->info);
    dense_info->in_dim0 = 5; // input_len
    dense_info->units = 64;  // output_units
    dense_info->weight = (float *)dense_w;
    dense_info->bias = (float *)dense_b;
    dense_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    // Layer 3: dense_1
    float *dense_1_output = (float *)malloc((32) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = dense_output;
    currLayer->dense_layer->output = dense_1_output;

    DenseInfo *dense_1_info = &(currLayer->dense_layer->info);
    dense_1_info->in_dim0 = 64; // input_len
    dense_1_info->units = 32;   // output_units
    dense_1_info->weight = (float *)dense_1_w;
    dense_1_info->bias = (float *)dense_1_b;
    dense_1_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    // Layer 4: dense_2
    float *dense_2_output = (float *)malloc((1) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = dense_1_output;
    currLayer->dense_layer->output = dense_2_output;

    DenseInfo *dense_2_info = &(currLayer->dense_layer->info);
    dense_2_info->in_dim0 = 32; // input_len
    dense_2_info->units = 1;    // output_units
    dense_2_info->weight = (float *)dense_2_w;
    dense_2_info->bias = (float *)dense_2_b;
    dense_2_info->act = ACTI_TYPE_LINEAR;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    // Layer 5: output_layer
    currLayer->type = LAYER_TYPE_OUTPUT;
    currLayer->name = strdup("Output");
    currLayer->output_layer = (OutputLayer *)malloc(sizeof(OutputLayer));
    currLayer->output_layer->output = dense_2_output;

    OutputInfo *output_info = &(currLayer->output_layer->info);
    output_info->dim0 = 1;
    currLayer->next = NULL;

    return headptr;
}
