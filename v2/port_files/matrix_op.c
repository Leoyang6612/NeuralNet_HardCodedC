#include "matrix_op.h"


// y = Wx
// W = (2, 15)
// x = (2,)
// y = (15,)
void vector_map(float *y, float *W, float *x, int size_in, int size_out, BufferState state_out)
{
    for (int i = 0; i < size_out; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < size_in; j++)
        {
            sum += W[j * size_out + i] * x[j];
            // it[i] += Xt[j] * W_i[j][i];
        }
        if (state_out == BUFFER_STATE_ACCUMULATE)
        {
            y[i] += sum;
        }
        else
        {
            y[i] = sum;
        }
    }
}

// y = w * x (len(w) == len(x))
void vector_pointwise_mul(float *y, float *w, float *x, int size, BufferState state_out)
{
    for (int i = 0; i < size; i++)
    {
        if (state_out == BUFFER_STATE_ACCUMULATE)
        {
            y[i] += w[i] * x[i];
        }
        else
        {
            y[i] = w[i] * x[i];
        }
    }
}

// y = x + b
void vector_pointwise_add(float *y, float *x, float *b, int size, BufferState state_out)
{
    for (int i = 0; i < size; i++)
    {
        if (state_out == BUFFER_STATE_ACCUMULATE)
        {
            y[i] += x[i] + b[i];
        }
        else
        {
            y[i] = x[i] + b[i];
        }
    }
}
