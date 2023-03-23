// ***ACTIVATION FUNCTIONS***
void ReLU(float *input, int input_len);
void Softmax(float *input, int input_len);
void HardSigmoid(float *input, uint8_t input_len);
void Sigmoid(float *input, uint8_t input_len);
void Tanh(float *input, uint8_t input_len);
void Activation_Func(ACTI_TYPE activation,
                     float *input,
                     int input_len);

// ***ARRAY MANIPULATION FUNCTIONS***
float *Arr_Copy(float *reference, uint8_t start, uint8_t len);
float *Arr_Plus(float *input1, float *input2, uint8_t len);
float *Arr_Multi(float *input1, float *input2, uint8_t len);

// ***BASIC NEURAL NET FUNCTIONS***
float Conv_ReLU(float *input,
                float *weight,
                float *bias,
                uint8_t kernel_size,
                uint8_t filter_depth,
                uint8_t filters_i,
                uint8_t len_index,
                uint8_t filters);
float *Conv1D(const uint8_t filters,
              const uint8_t kernel_size,
              const uint8_t filter_depth,
              const uint8_t input_len,
              float *input,
              float *weight,
              float *bias);
float *AveragePooling1D(float *input,
                        uint8_t pool_size,
                        uint8_t input_len,
                        uint8_t input_depth);
float *Dense(float *input,
             float *weight,
             float *bias,
             uint8_t input_len,
             uint8_t output_len,
             ACTI_TYPE activation);
float *LSTM_Conv(float *input,
                 float *weight,
                 uint8_t input_len,
                 uint8_t output_len,
                 int gate_index);
void LSTM_Bias(float *input,
               float *bias,
               uint8_t len,
               uint8_t gate_index);
float *Lstm(float *input,
            float *in_weight,
            float *re_weight,
            float *bias,
            uint8_t time_step,
            uint8_t input_len,
            uint8_t output_len,
            bool return_seq);

float *Standardize(float *input, uint8_t len, uint8_t depth);

float *Normalize(float *input, uint8_t len, uint8_t depth);

void Next_Layer_Input(LAYER *p, float *input);

void Input_Layer(LAYER *p);
void Conv1D_Layer(LAYER *p);
void Average_Pooling_1D_Layer(LAYER *p);
void Dense_Layer(LAYER *p);
void LSTM_Layer(LAYER *p);