// EXAMPLE CODE FOR NN INFERENCE
// by Young Boy, 08.23.2022
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer_structure.h"
#include "model.h"
#include <time.h>

#define TEST_TIMES 1000000
#define AVG_TIMES 10
void timeFunction(void (*function)())
{
    clock_t start, end;
    double cpu_time_used_average = 0.0f;
    double cpu_time_used;
    for (int i = 0; i < AVG_TIMES; i++)
    {
        start = clock();
        function();
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        cpu_time_used_average += cpu_time_used;
        printf("The function took %f seconds to execute.\n", cpu_time_used);
    }
    cpu_time_used_average /= AVG_TIMES;
    printf("The function took %f seconds to execute.(%d in average)\n", cpu_time_used_average, AVG_TIMES);
}

void myFunction()
{
    float test_input[60] = {0.};
    for (int i = 0; i < 60; i++)
    {
        test_input[i] = 60 + 2 * i;
        // printf("%f ", test_input[i]);
    }
    // printf("\n");
    LAYER *myModel = load_model();
    for (int i = 0; i < TEST_TIMES; i++)
    {
        unsigned int ret = predict(test_input, myModel);
        // printf("class: %u\n", ret);
    }
}

int main()
{
    timeFunction(myFunction);
    return 0;
}
