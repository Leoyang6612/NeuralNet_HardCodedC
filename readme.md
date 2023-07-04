# Project: NN-on-C (Neural Network Inference on C code)

**v1.0 Author:** 吳鑑軒 & NeatLab學長

**v2.0 Author:** 楊博禕

## How to use?

**Step 0:** Train the model in a Python environment and move it into the "model" folder.

**Step 1:** Run `makeheader_v2.py` with Python3 to generate `model.c`, `model.h`, `weight.c`, and `weight.h`.

**Step 2:** Port all the `.c` and `.h` files into your project, and include `#include "layer_struct.h"` and `#include "model.h"` in your `main.c` header.

**Step 3:** Call `predict()` and `load_model()` functions in your C code (refer to `main.c` for an example).

**Step 4:** That's all!

## Debugging

In `model_forward.c` file, uncomment the following lines to enable debugging:
```c
#define PRINT_LAYER_SHAPE
#define PRINT_LAYER_OUTPUT
```

## VERSION 1
### v1.0, 08.23.2022, young boy

- Dense Layer, Output Layer ✓
- Modified `Normalize-mymodel_API.c`
- Use `enum` instead of `char[]` for `LAYER_TYPE` and `ACTI_TYPE`, refer to `layer_structure.h`
- Check memory leakage using Dr. Memory (except for the entire model not being freed)
  > thanks a lot to [this link](https://www.cnblogs.com/phpandmysql/p/10953058.html)
- Improved readability by adding string interpolation to `makeheader.py`

## VERSION 2
### v2.0, 03.22.2023, young boy

- Modified `makeheader.py` to `makeheader_v2.py`
- Added support for layers: Conv1D, Dense, Lstm
- Added support for activation functions: ReLU, softmax, sigmoid, hardsigmoid, tanh, linear

## VERSION 2.1
### v2.1, 04.27.2023, young boy

- Added `ARM CMSIS-DSP lib`
- Support for Dense layer only

## VERSION 2.2
### v2.2, 07.04.2023, 劉敏彤

- Fixed bug in `LSTM_forward` by adding activation function