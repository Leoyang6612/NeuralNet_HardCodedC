Neural Network Inference

作者：吳鑑軒 

修改：楊博禕 v1.0

### v1.0, 08.23, young boy
>Dense Layer, Output Layer 確定ok

>Normalize-mymodel_API.c 修正

>用enum取代char[]：LAYER_TYPE & ACTI_TYPE，詳見layer_structure.h

>drmemory檢查 memory leakage (剩下整個model沒有free) 
>>感謝 https://www.cnblogs.com/phpandmysql/p/10953058.html

>makeheader.py 加入string interpolation，增加可讀性
