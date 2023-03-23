#ifndef LAYER_STRUCTURE_H
#define LAYER_STRUCTURE_H

#include <stdint.h>
#include <stdbool.h>

typedef enum e_layer_type
{
    LAYER_TYPE_INPUT,
    LAYER_TYPE_OUTPUT,
    LAYER_TYPE_CONV1D,
    LAYER_TYPE_AVERAGE_POOL1D,
    LAYER_TYPE_DENSE,
    LAYER_TYPE_LSTM
} LAYER_TYPE;

typedef enum e_activation_type
{
    ACTI_TYPE_RELU,
    ACTI_TYPE_SOFTMAX,
    ACTI_TYPE_SIGMOID,
    ACTI_TYPE_HARDSIGMOID,
    ACTI_TYPE_TANH,
} ACTI_TYPE;

typedef struct conv1d_s
{
    uint8_t filters;
    uint8_t kernel_size;
    uint8_t filter_depth;
    uint8_t input_len;
    float *input;
    float *weight;
    float *bias;
} CONV1D_LAYER;

typedef struct dense_s
{
    float *input;
    float *weight;
    float *bias;
    uint8_t input_len;
    uint8_t output_len;
    ACTI_TYPE activation;
} DENSE_LAYER;

typedef struct average_pooling_1d_s
{
    float *input;
    uint8_t pool_size;
    uint8_t input_len;
    uint8_t input_depth;
} AVG_POOL1D_LAYER;

typedef struct lstm_s
{
    float *input;
    float *recurrent;
    float *memarr;
    float *input_weight;
    float *recurrent_weight;
    float *bias;
    uint8_t time_step;
    uint8_t input_len;
    uint8_t output_len;
    bool return_seq;
} LSTM_LAYER;

typedef struct input_s
{
    float *input;
    uint8_t len;
    uint8_t depth;
} INPUT_LAYER;

typedef struct output_s
{
    float *output;
    uint8_t units;
} OUTPUT_LAYER;

typedef struct layer_s
{
    char *name;
    LAYER_TYPE type;
    INPUT_LAYER *input_layer;
    CONV1D_LAYER *conv1d_layer;
    AVG_POOL1D_LAYER *avg_pool1D_layer;
    DENSE_LAYER *dense_layer;
    LSTM_LAYER *lstm_layer;
    OUTPUT_LAYER *output_layer;
    void (*func)(struct layer_s *);
    struct layer_s *next_layer;
} LAYER;

#endif