#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "layer_structure.h"
#include "create_layer_API.h"
#include "mymodel_API.h"
void Create_Input_Layer(LAYER *layer, uint8_t len, uint8_t depth)
{
    layer->input_layer = (InputLayer *)malloc(sizeof(InputLayer));
    layer->input_layer->len = len;
    layer->input_layer->depth = depth;
    layer->func = Input_Layer;
    layer->name = strdup("InputLayer");
    layer->type = LAYER_TYPE_INPUT;
    layer->next_layer = (LAYER *)malloc(sizeof(LAYER));
    return;
}

void Create_Conv1D_Layer(
    LAYER *layer,
    uint8_t filters,
    uint8_t kernel_size,
    uint8_t filter_depth,
    uint8_t input_len,
    float *weight,
    float *bias)
{
    layer->conv1d_layer = (CONV1D_LAYER *)malloc(sizeof(CONV1D_LAYER));
    layer->conv1d_layer->bias = bias;
    layer->conv1d_layer->weight = weight;
    layer->conv1d_layer->filters = filters;
    layer->conv1d_layer->kernel_size = kernel_size;
    layer->conv1d_layer->filter_depth = filter_depth;
    layer->conv1d_layer->input_len = input_len;
    layer->func = Conv1D_Layer;
    layer->name = strdup("Conv1D");
    layer->type = LAYER_TYPE_CONV1D;
    layer->next_layer = (LAYER *)malloc(sizeof(LAYER));
    return;
}

void Create_Average_Pooling_1D_Layer(
    LAYER *layer,
    uint8_t pool_size,
    uint8_t input_len,
    uint8_t input_depth)
{
    layer->avg_pool1D_layer = (AVG_POOL1D_LAYER *)malloc(sizeof(AVG_POOL1D_LAYER));
    layer->avg_pool1D_layer->pool_size = pool_size;
    layer->avg_pool1D_layer->input_len = input_len;
    layer->avg_pool1D_layer->input_depth = input_depth;
    layer->func = Average_Pooling_1D_Layer;
    layer->name = strdup("AveragePolling1D");
    layer->type = LAYER_TYPE_AVERAGE_POOL1D;
    layer->next_layer = (LAYER *)malloc(sizeof(LAYER));
    return;
}

void Create_Dense_Layer(
    LAYER *layer,
    float *weight,
    float *bias,
    uint8_t input_len,
    uint8_t output_len,
    ACTI_TYPE activation)
{
    layer->dense_layer = (DENSE_LAYER *)malloc(sizeof(DENSE_LAYER));
    layer->dense_layer->weight = weight;
    layer->dense_layer->bias = bias;
    layer->dense_layer->input_len = input_len;
    layer->dense_layer->output_len = output_len;
    layer->dense_layer->activation = activation;
    layer->func = Dense_Layer;
    layer->name = strdup("Dense");
    layer->type = LAYER_TYPE_DENSE;
    layer->next_layer = (LAYER *)malloc(sizeof(LAYER));
    return;
}

void Create_LSTM_Layer(
    LAYER *layer,
    float *input_weight,
    float *recurrent_weight,
    float *bias,
    uint8_t time_step,
    uint8_t input_len,
    uint8_t output_len,
    bool return_seq)
{
    layer->lstm_layer = (LSTM_LAYER *)malloc(sizeof(LSTM_LAYER));
    layer->lstm_layer->recurrent = (float *)calloc(output_len, sizeof(float));
    layer->lstm_layer->memarr = (float *)calloc(output_len, sizeof(float));
    layer->lstm_layer->input_weight = input_weight;
    layer->lstm_layer->recurrent_weight = recurrent_weight;
    layer->lstm_layer->bias = bias;
    layer->lstm_layer->time_step = time_step;
    layer->lstm_layer->input_len = input_len;
    layer->lstm_layer->output_len = output_len;
    layer->lstm_layer->return_seq = return_seq;
    layer->func = LSTM_Layer;
    layer->name = strdup("LSTM");
    layer->type = LAYER_TYPE_LSTM;
    layer->next_layer = (LAYER *)malloc(sizeof(LAYER));
    return;
}

void Create_Output_Layer(LAYER *layer, uint8_t units)
{
    layer->output_layer = (OUTPUT_LAYER *)malloc(sizeof(OUTPUT_LAYER));
    layer->output_layer->units = units;
    layer->name = strdup("Output");
    layer->type = LAYER_TYPE_OUTPUT;
    layer->next_layer = NULL;
    return;
}