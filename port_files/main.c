// EXAMPLE CODE FOR NN INFERENCE
// by Young Boy, 08.23.2022
#include <stdio.h>
#include <stdlib.h>
#include "layer_structure.h"
#include "model.h"
#include <string.h>

int main()
{
    float test_input[60] = {0.};
    for (int i = 0; i < 60; i++)
    {
        test_input[i] = 60 + 2 * i;
    }

    LAYER *dnn_model = load_model();
    unsigned int ret = predict(test_input, dnn_model);
    printf("ret: %u\n", ret);
    // system("pause");
    return 0;
}