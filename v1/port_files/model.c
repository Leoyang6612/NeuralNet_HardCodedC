
#include <stdio.h>
#include <stdlib.h>
#include "layer_structure.h"
#include "mymodel_API.h"
#include "create_layer_API.h"
#include "header.h"

LAYER *load_model()
{
    LAYER *layer, *layer_head;
    layer = (LAYER *)malloc(sizeof(LAYER));
    layer_head = layer;
    Create_Input_Layer(layer, 60, 1);
    layer = layer->next_layer;

    Create_Dense_Layer(layer, (float *)densew, denseb, 60, 16, denseac);
    layer = layer->next_layer;

    Create_Dense_Layer(layer, (float *)dense_1w, dense_1b, 16, 32, dense_1ac);
    layer = layer->next_layer;

    Create_Dense_Layer(layer, (float *)dense_2w, dense_2b, 32, 16, dense_2ac);
    layer = layer->next_layer;

    Create_Dense_Layer(layer, (float *)dense_3w, dense_3b, 16, 2, dense_3ac);
    layer = layer->next_layer;

    Create_Output_Layer(layer, 2);
    return layer_head;
}

unsigned int predict(float *input, LAYER *head)
{
    LAYER *layer = head;
    layer->input_layer->input = input;
    // printf("======== Inference  Start ========\r\n");
    while (layer->type != LAYER_TYPE_OUTPUT)
    {
        layer->func(layer);
        layer = layer->next_layer;
    }

    int units = layer->output_layer->units;
    float *output = layer->output_layer->output;

    unsigned int ret = 0;
    float max_confidence = output[0];
    // printf("%.6f\r\n", output[0]);

    for (int i = 1; i < units; i++)
    {
        if (output[i] > max_confidence)
        {
            max_confidence = output[i];
            ret = i;
        }
        // printf("%.6f\r\n", output[i]);
    }
    // printf("======== Inference Finish ========\r\n");
    free(output);
    return ret;
}
