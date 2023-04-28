## Project: Neural Network Inference on C code

>v1.0 Author：吳鑑軒 & NeatLab學長

>v2.0 Author：楊博禕

## How to use?
>1. run makeheader_v2.py with Python3 => generate model.c weight.h
>2. include all the .c .h files into your project
>3. call predict() and load_model() functions in your c code (see main.c for example)
>4. That's all!

## VERSION 1
### v1.0, 08.23.2022, young boy
>Dense Layer, Output Layer ✓ 

>Normalize-mymodel_API.c 修正

>用enum取代char[]：LAYER_TYPE & ACTI_TYPE，詳見layer_structure.h

>drmemory檢查 memory leakage (剩下整個model沒有free) 
>>特別感謝 https://www.cnblogs.com/phpandmysql/p/10953058.html

>makeheader.py 加入string interpolation，增加可讀性

## VERSION 2
### v2.0, 03.22.2023, young boy
>modified: makeheader.py => makeheader_v2.py

>Support layer: Conv1D, Dense, Lstm

>Support activation function: ReLU, softmax, sigmoid, hardsigmoid, tanh, linear

## VERSION 2.1
### v2.0, 04.27.2023, young boy
>ARM CMSIS-DSP lib

>Support layer: Dense only
