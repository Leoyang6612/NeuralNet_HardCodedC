#include "layer_struct.h"
#include "model_forward_arm_op.h"
#include "acti_forward.h"
#include "arm_math.h"
#include "matrix_op.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nrf_delay.h"

//#define PRINT_LAYER_SHAPE
//#define PRINT_LAYER_OUTPUT

void normalize_forward_arm_op(InputLayer *layer) {

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

    //get min & max of depth(features)
    memcpy(min_arr, input, input_depth * sizeof(float));
    memcpy(max_arr, input, input_depth * sizeof(float));

#ifdef PRINT_LAYER_SHAPE
    printf("Now normalize...\r\n");
    printf("(%d, %d) => (%d, %d)\r\n", input_length, input_depth, input_length, input_depth);
#endif

    for (int i = 1; i < input_length; i++) {
	for (int j = 0; j < input_depth; j++) {
	    float curr_val = input[i * input_depth + j];
	    min_arr[j] = (min_arr[j] > curr_val) ? curr_val : min_arr[j];
	    max_arr[j] = (max_arr[j] < curr_val) ? curr_val : max_arr[j];
	}
    }

    arm_sub_f32(max_arr, min_arr, max_arr, input_depth);
    for (int i = 0; i < input_length; i++) {
	// void arm_sub_f32(const float32_t *pSrcA, const float32_t *pSrcB, float32_t *pDst, uint32_t blockSize);
	arm_sub_f32(&input[i * input_depth], min_arr, &input[i * input_depth], input_depth);

	// max_arr now: max - min
	for (int j = 0; j < input_depth; j++) {
	    output[i * input_depth + j] = input[i * input_depth + j] / max_arr[j];
	}
    }

    free(min_arr);
    free(max_arr);

#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < input_length; i++) {
	for (int j = 0; j < input_depth; j++) {
	    printf("%f ", output[i * input_depth + j]);
	}
	printf("\n");
    }
#endif
}

void dense_forward_arm_op(DenseLayer *layer) {
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

    // create input matrix W and vector x
    arm_matrix_instance_f32 mat_w = {input_length, output_length, weight};
    arm_matrix_instance_f32 vec_x = {input_length, 1, input};
    
    // create output vector y
    arm_matrix_instance_f32 vec_y = {output_length, 1, output};
    
    // create temporary buffer
    arm_matrix_instance_f32 mat_buffer = {output_length, 1, output};

    // perform matrix multiplication
    arm_mat_mult_f32(&mat_w, &vec_x, &mat_buffer);
    
    // y = Wx
    //vector_map(output, weight, input, input_length, output_length, BUFFER_STATE_OVERWRITE);
    acti_forward(act, output, output_length);


#ifdef PRINT_LAYER_OUTPUT
    for (int i = 0; i < output_length; i++) {
	printf("%.4f ", output[i]);
    }
    printf("\n");
#endif
}