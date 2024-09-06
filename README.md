# DRL-ice-hockey

The repository contains explanation about the network structure of paper "[Deep Learning-based Pass Intent Prediction in Football Matche](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11862501)".  


## Network Structure:  

| name        | nodes           | activation function  |
| ------------- |:-------------:| -----:|
| Input Embeding    | 512           | N/A |
| Position Encoding| 1024     |  Relu |
| Temporal Transformer | 1000      |  Relu |
| Spatial Transformer | 3      |  N/A |
| Temporal Transformer | 3      |  N/A |


## Image of network structure:  

<img src=./images/pass_intention_algorithm.png alt="drawing" height="320" width="420"/>