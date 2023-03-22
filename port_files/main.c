// EXAMPLE CODE FOR NN INFERENCE
// by Young Boy, 08.23.2022
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer_struct.h"
#include "model.h"

int main()
{
    float test_input[60] = {0.};
    for (int i = 0; i < 60; i++)
    {
        test_input[i] = 60 + 2 * i;
        // printf("%f ", test_input[i]);
    }
    // printf("\n");

    Layer *myModel = load_model();
    unsigned int ret = predict(myModel, test_input);
    printf("class: %u\n", ret);
    return 0;
}