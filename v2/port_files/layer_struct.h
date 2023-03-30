#ifndef _LAYER_STRUCT_H_
#define _LAYER_STRUCT_H_

#include <stdbool.h>

typedef enum
{
    LAYER_TYPE_INPUT,
    LAYER_TYPE_OUTPUT,
    LAYER_TYPE_CONV1D,
    LAYER_TYPE_DENSE,
    LAYER_TYPE_LSTM,
    LAYER_TYPE_AVERAGE_POOL1D,
    LAYER_TYPE_MAX_POOL1D
} LayerType;

typedef enum
{
    ACTI_TYPE_NO_ACTI,
    ACTI_TYPE_RELU,
    ACTI_TYPE_SOFTMAX,
    ACTI_TYPE_SIGMOID,
    ACTI_TYPE_HARDSIGMOID,
    ACTI_TYPE_TANH
} ActivationType;

typedef struct
{
    int dim0; // input_len
    int dim1; // depth
    bool normalize;
} InputInfo;

typedef struct
{
    int dim0; // output_len
} OutputInfo;

typedef struct
{
    int in_dim0; // input_len
    int in_dim1; // depth
    int filters;
    int kernel_size;
    int padding;
    int stride;

    float *weight;
    float *bias;

    ActivationType act;
} Conv1dInfo;

typedef struct
{
    int in_dim0; // input_len
    int units;   // output_units

    float *weight;
    float *bias;

    ActivationType act;
} DenseInfo;

typedef struct
{
    // input_shape(timestep, input_dim)
    // output_shape(timestep, input_dim)
    int in_dim0; // time_step
    int in_dim1; // input_dim
    int units;   // == output_len if (return_seq == false)

    // Ex:
    // input_shape = (20, 3)
    // units = 16
    // output_shape = (16,)         return_seq == false
    // output_shape = (20, 16)      return_seq == true
    

    // input_weight (input_dim, 4 * units)
    // input gate
    float *weight_i;
    float *rcr_weight_i;
    float *bias_i;

    // forget gate
    float *weight_f;
    float *rcr_weight_f;
    float *bias_f;

    // new candidate cell state: Ct_bar
    float *weight_c;
    float *rcr_weight_c;
    float *bias_c;

    // output gate
    float *weight_o;
    float *rcr_weight_o;
    float *bias_o;

    bool return_seq;
} LstmInfo;

typedef struct
{
    int in_dim0; // input_len
    int in_dim1; // depth
    int pool_size;
} AvgPool1dInfo;

typedef struct
{
    int in_dim0; // input_len
    int in_dim1; // depth
    int pool_size;
} MaxPool1dInfo;

typedef struct Conv1dLayer_t
{
    Conv1dInfo info;
    void (*exec)(struct Conv1dLayer_t *);
    float *input;
    float *output;
} Conv1dLayer;

typedef struct DenseLayer_t
{
    DenseInfo info;
    void (*exec)(struct DenseLayer_t *);
    float *input;
    float *output;
} DenseLayer;

typedef struct LstmLayer_t
{
    LstmInfo info;
    void (*exec)(struct LstmLayer_t *);
    float *input;
    float *output;
    float *Xt;
    float *it;
    float *ht;
    float *ht_1;
    float *ft;
    float *Ct;
    float *Ct_1;
    float *Ct_bar;
    float *Ot;
} LstmLayer;

typedef struct AvgPool1dLayer_t
{
    AvgPool1dInfo info;
    void (*exec)(struct AvgPool1dLayer_t *);
    float *input;
    float *output;
} AvgPool1dLayer;

typedef struct MaxPool1dLayer_t
{
    MaxPool1dInfo info;
    void (*exec)(struct MaxPool1dLayer_t *);
    float *input;
    float *output;
} MaxPool1dLayer;

typedef struct InputLayer_t
{
    InputInfo info;
    void (*exec)(struct InputLayer_t *);
    float *input;
    float *output;
} InputLayer;

typedef struct OutputLayer_t
{
    OutputInfo info;
    float *output;
} OutputLayer;

typedef struct Layer_t
{
    char *name;
    LayerType type;
    InputLayer *input_layer;
    OutputLayer *output_layer;
    Conv1dLayer *conv1d_layer;
    DenseLayer *dense_layer;
    LstmLayer *lstm_layer;
    AvgPool1dLayer *avg_pool1d_layer;
    MaxPool1dLayer *max_pool1d_layer;
    struct Layer_t *next;
} Layer;

#endif