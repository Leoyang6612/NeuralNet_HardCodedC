## Project: Neural Network Inference on C code

>Author：吳鑑軒 & NeatLab學長

>Modified：楊博禕 v1.0, v2.0

## How to use?
>1. run makeheader_v2.py with Python3 => generate model.c model.h
>2. include all the .c .h files into your project
>3. call predict() and load_model() functions in your c code
>4. That's all!

## VERSION
### v1.0, 08.23.2022, young boy
>Dense Layer, Output Layer ✓ 

>Normalize-mymodel_API.c 修正

>用enum取代char[]：LAYER_TYPE & ACTI_TYPE，詳見layer_structure.h

>drmemory檢查 memory leakage (剩下整個model沒有free) 
>>特別感謝 https://www.cnblogs.com/phpandmysql/p/10953058.html

>makeheader.py 加入string interpolation，增加可讀性

## VERSION
### v2.0, 03.22.2023, young boy
>modified: makeheader.py => makeheader_v2.py

>Support layer: Conv1D, Dense

>Support activation function: ReLU, softmax, sigmoid, hardsigmoid, tanh
