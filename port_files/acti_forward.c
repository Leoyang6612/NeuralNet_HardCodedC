#include <stdio.h>
#include <math.h>
#include "layer_struct.h"
#include "acti_forward.h"

static void ReLU(float *input, int input_len)
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

static void Softmax(float *input, int input_len)
{
    // find max in input to prevent exp ovreflow!
    float max = input[0];
    for (int i = 1; i < input_len; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    double base = 0.0f;
    for (int i = 0; i < input_len; i++)
    {
        input[i] = exp(input[i] - max);
        base += exp(input[i]);
    }

    // printf("Base: %f\n", base);
    // for (int i = 0; i < input_len; i++)
    // {
    //     input[i] = (float)exp(input[i]) / base;
    //     printf("%f\n", input[i]);
    // }
}

// LSTM default recurrent activation function in Keras
static void HardSigmoid(float *input, int input_len)
{
    int len_i;
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
}

static void Sigmoid(float *input, int input_len)
{
    int len_i;
    for (len_i = 0; len_i < input_len; len_i++)
    {
        input[len_i] = 1 / (1 + exp(-1 * input[len_i]));
    }
    return;
}

static void Tanh(float *input, int input_len)
{
    int len_i;
    for (len_i = 0; len_i < input_len; len_i++)
    {
        input[len_i] = (float)tanh(input[len_i]);
    }
}

void acti_forward(ActivationType activation, float *input, int input_len)
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
        printf("Acti:%d undefined!\n", activation);
        break;
    }
}