#ifndef _MAXTRIX_OP_H_
#define _MAXTRIX_OP_H_

typedef enum
{
    BUFFER_STATE_ACCUMULATE,
    BUFFER_STATE_OVERWRITE
} BufferState;

void vector_map(float *, float *, float *, int, int, BufferState);
void vector_pointwise_mul(float *, float *, float *, int, BufferState);
void vector_pointwise_add(float *, float *, float *, int, BufferState);

#endif