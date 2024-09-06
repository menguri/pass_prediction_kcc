# Transformer : Soccer pass-receiver prediction

The repository contains explanation about the network structure of paper "[Deep Learning-based Pass Intent Prediction in Football Matche](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11862501)".  


<br>

## Network Structure:  

| Start        | layer           |
| ------------- |:-------------:|
| 1. Input Embeding    | x           |
| 2. Position Encoding| 6     |

<br>
 
| Middle        | Attention           |
| ------------- |:-------------:|
| 3. Temporal Transformer | 8     |
| 4. Spatial Transformer | 8      |
| 5. Temporal Transformer | 8      |

<br>

| End        | output           |
| ------------- |:-------------:|
| 6. Decoder | 11      |
| 7. Softmax | 11      |

<br>

## Image of network structure:  

<img src=./images/pass_intention_algorithm.png alt="drawing" width="420"/>

<br>

## Preprocess metrica-data
For running this model, you should make your data fit with specific format. <br>
Check the link(https://github.com/GunHeeJoe/metrica_preprocess). You can check the format and get some tips for preprocessing metrica-data of statsbomb.

<br>

## Animation of Unsuccess passes for labeling Intended-receiver wiht metrica-data preprocessed

[Metrica Pass Animation](https://drive.google.com/drive/folders/1rKPn8ivr99hSjokezpy_jRRW8u8ZjAZx?usp=sharing)
[Metrica Data](https://drive.google.com/drive/u/0/folders/1VoLVWMaFLXY-8KVkLL9Z6NGLYsMP23y5)

<br>

## About Notebook

Notebook folder holds some baseline notebooks which show 'how to get metrica data', 'how to process basic experiments in paper', 'how to distinguish events'. Unfortunately we can't upload the model for copyright issues. If you want to get details about model, Just send me Email(tatalintelli@gmail.com).

<br>
Good Luck!
