#ifndef _MODEL_FORWARD_H_
#define _MODEL_FORWARD_H_

void normalize_forward(InputLayer *);
void conv1d_forward(Conv1dLayer *);
void dense_forward(DenseLayer *);
void lstm_forward(LstmLayer *);
void avg_pool1d_forward(AvgPool1dLayer *);
void max_pool1d_forward(MaxPool1dLayer *);

#endif