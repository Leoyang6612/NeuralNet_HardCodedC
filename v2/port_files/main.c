// EXAMPLE CODE FOR NN INFERENCE
// by Young Boy, 08.23.2022
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "layer_struct.h"
#include "model.h"

#define TEST_TIMES 1
#define AVG_TIMES 10

Layer *myModel = NULL;
float test_input[60] = {0.};

void timeFunction(void (*function)())
{
    struct timespec start, end;
    double cpu_time_used_average = 0.0;
    double cpu_time_used_for_each_predict;
    for (int i = 0; i < AVG_TIMES; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        function();
        clock_gettime(CLOCK_MONOTONIC, &end);
        cpu_time_used_for_each_predict = ((double)(end.tv_sec - start.tv_sec) * 1e6 +
                                          (double)(end.tv_nsec - start.tv_nsec) * 1e-3);
        cpu_time_used_for_each_predict /= TEST_TIMES;
        cpu_time_used_average += cpu_time_used_for_each_predict;
        printf("The function took %.4f microseconds for each prediction.\n", cpu_time_used_for_each_predict);
    }
    cpu_time_used_average /= AVG_TIMES;
    printf("The function took %.4f microseconds to execute (%d times in average).\n", cpu_time_used_average, AVG_TIMES);
}

void myFunction()
{
    for (int i = 0; i < TEST_TIMES; i++)
    {
        unsigned int ret = predict(myModel, test_input);
        // printf("class: %u\n", ret);
    }
}

int main()
{
    for (int i = 0; i < 60; i++)
    {
        test_input[i] = 60 + 2 * i;
        // printf("%f ", test_input[i]);
    }
    // printf("\n");
    myModel = load_model();
    timeFunction(myFunction);
    return 0;
}
