# Transformer : Soccer pass-receiver prediction

The repository contains explanation about the network structure of paper "[Deep Learning-based Pass Intent Prediction in Football Matche](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11862501)".  


## Network Structure:  

| 1. start        | layer           |
| ------------- |:-------------:|
| Input Embeding    | x           |
| Position Encoding| 6     |

| 2. middle        | Attention           |
| Temporal Transformer | 8     |
| Spatial Transformer | 8      |
| Temporal Transformer | 8      |

| 3. end        | output           |
| Decoder | 11      |
| Softmax | 11      |

<br>
<br>

## Image of network structure:  

<img src=./images/pass_intention_algorithm.png alt="drawing"/>


<br>
<br>


## Preprocess metrica-data

