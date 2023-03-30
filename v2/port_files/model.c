
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer_struct.h"
#include "weight.h"
#include "model.h"
#include "model_forward.h"

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

    
    // Layer 2: lstm_2
    float *lstm_2_output = (float *)malloc((20 * 16) * sizeof(float));

    currLayer->type = LAYER_TYPE_LSTM;
    currLayer->name = strdup("Lstm");
    currLayer->lstm_layer = (LstmLayer *)malloc(sizeof(LstmLayer));
    currLayer->lstm_layer->exec = lstm_forward;
    currLayer->lstm_layer->input = input;
    currLayer->lstm_layer->output = lstm_2_output;

    currLayer->lstm_layer->Xt = (float *)calloc(3, sizeof(float));
    currLayer->lstm_layer->it = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ht = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ht_1 = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ft = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct_1 = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct_bar = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ot = (float *)calloc(16, sizeof(float));

    LstmInfo *lstm_2_info = &(currLayer->lstm_layer->info);
    lstm_2_info->in_dim0 = 20; // time_step
    lstm_2_info->in_dim1 = 3;  // features
    lstm_2_info->units = 16;
    lstm_2_info->return_seq = true;
    // input gate
    lstm_2_info->weight_i = (float *)lstm_2_w_i;
    lstm_2_info->rcr_weight_i = (float *)lstm_2_u_i;
    lstm_2_info->bias_i = (float *)lstm_2_b_i;
    // forget gate
    lstm_2_info->weight_f = (float *)lstm_2_w_f;
    lstm_2_info->rcr_weight_f = (float *)lstm_2_u_f;
    lstm_2_info->bias_f = (float *)lstm_2_b_f;
    // new candidate cell state: Ct_bar
    lstm_2_info->weight_c = (float *)lstm_2_w_c;
    lstm_2_info->rcr_weight_c = (float *)lstm_2_u_c;
    lstm_2_info->bias_c = (float *)lstm_2_b_c;
    // output gate
    lstm_2_info->weight_o = (float *)lstm_2_w_o;
    lstm_2_info->rcr_weight_o = (float *)lstm_2_u_o;
    lstm_2_info->bias_o = (float *)lstm_2_b_o;

    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 3: lstm_3
    float *lstm_3_output = (float *)malloc((1 * 16) * sizeof(float));

    currLayer->type = LAYER_TYPE_LSTM;
    currLayer->name = strdup("Lstm");
    currLayer->lstm_layer = (LstmLayer *)malloc(sizeof(LstmLayer));
    currLayer->lstm_layer->exec = lstm_forward;
    currLayer->lstm_layer->input = lstm_2_output;
    currLayer->lstm_layer->output = lstm_3_output;

    currLayer->lstm_layer->Xt = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->it = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ht = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ht_1 = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->ft = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct_1 = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ct_bar = (float *)calloc(16, sizeof(float));
    currLayer->lstm_layer->Ot = (float *)calloc(16, sizeof(float));

    LstmInfo *lstm_3_info = &(currLayer->lstm_layer->info);
    lstm_3_info->in_dim0 = 20; // time_step
    lstm_3_info->in_dim1 = 16;  // features
    lstm_3_info->units = 16;
    lstm_3_info->return_seq = false;
    // input gate
    lstm_3_info->weight_i = (float *)lstm_3_w_i;
    lstm_3_info->rcr_weight_i = (float *)lstm_3_u_i;
    lstm_3_info->bias_i = (float *)lstm_3_b_i;
    // forget gate
    lstm_3_info->weight_f = (float *)lstm_3_w_f;
    lstm_3_info->rcr_weight_f = (float *)lstm_3_u_f;
    lstm_3_info->bias_f = (float *)lstm_3_b_f;
    // new candidate cell state: Ct_bar
    lstm_3_info->weight_c = (float *)lstm_3_w_c;
    lstm_3_info->rcr_weight_c = (float *)lstm_3_u_c;
    lstm_3_info->bias_c = (float *)lstm_3_b_c;
    // output gate
    lstm_3_info->weight_o = (float *)lstm_3_w_o;
    lstm_3_info->rcr_weight_o = (float *)lstm_3_u_o;
    lstm_3_info->bias_o = (float *)lstm_3_b_o;

    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 4: dense_3
    float *dense_3_output = (float *)malloc((16) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = lstm_3_output;
    currLayer->dense_layer->output = dense_3_output;

    DenseInfo *dense_3_info = &(currLayer->dense_layer->info);
    dense_3_info->in_dim0 = 16;      // input_len
    dense_3_info->units = 16;       // output_units
    dense_3_info->weight = (float *)dense_3_w;
    dense_3_info->bias = (float *)dense_3_b;
    dense_3_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 5: dense_4
    float *dense_4_output = (float *)malloc((32) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = dense_3_output;
    currLayer->dense_layer->output = dense_4_output;

    DenseInfo *dense_4_info = &(currLayer->dense_layer->info);
    dense_4_info->in_dim0 = 16;      // input_len
    dense_4_info->units = 32;       // output_units
    dense_4_info->weight = (float *)dense_4_w;
    dense_4_info->bias = (float *)dense_4_b;
    dense_4_info->act = ACTI_TYPE_RELU;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 6: dense_5
    float *dense_5_output = (float *)malloc((2) * sizeof(float));
    currLayer->type = LAYER_TYPE_DENSE;
    currLayer->name = strdup("Dense");
    currLayer->dense_layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    currLayer->dense_layer->exec = dense_forward;
    currLayer->dense_layer->input = dense_4_output;
    currLayer->dense_layer->output = dense_5_output;

    DenseInfo *dense_5_info = &(currLayer->dense_layer->info);
    dense_5_info->in_dim0 = 32;      // input_len
    dense_5_info->units = 2;       // output_units
    dense_5_info->weight = (float *)dense_5_w;
    dense_5_info->bias = (float *)dense_5_b;
    dense_5_info->act = ACTI_TYPE_SOFTMAX;
    currLayer->next = (Layer *)malloc(sizeof(Layer));
    currLayer = currLayer->next;

    
    // Layer 7: output_layer
    currLayer->type = LAYER_TYPE_OUTPUT;
    currLayer->name = strdup("Output");
    currLayer->output_layer = (OutputLayer *)malloc(sizeof(OutputLayer));
    currLayer->output_layer->output = dense_5_output;

    OutputInfo *output_info = &(currLayer->output_layer->info);
    output_info->dim0 = 2;
    currLayer->next = NULL;

    return headptr;
}
    