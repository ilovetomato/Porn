# Porn: Research on Wearable Human Activity Recognition  Based on Information Interaction Between Joint  Points

This repository contains the code I developed for my Bachelor 's Degree in Electronic Information thesis.

**Research on Wearable Human Activity Recognition  Based on Information Interaction Between Joint  Points**

**Abstract**: Aiming at the feature fusion of  multi-sensor data, this paper proposes a wearable human activity recognition model (Porn)  based on information interaction between joints. 

When **modeling the feature extraction of a single position sensor**, a multi-modal feature  fusion module, a temporal feature compression module, and a temporal feature extraction  module based on convolutional neural networks and recurrent neural networks are proposed.  By inputting the multi-modal sensor data of a single location into the above module, the  multi-modal feature and time series feature of the location sensor are extracted, and the  high-level semantic feature expression of the location sensor is finally obtained. 

<img src="IMG\single_sensor.png" style="zoom: 33%;" />

When **modeling the feature fusion of multiple position sensors**, an encoder and decoder  structure based on graph convolutional neural network and attention mechanism is proposed.  First of all, the sensor at each position is regarded as a node, and the high-level semantic  feature expression of each position sensor is regarded as the feature of the node. Secondly,  the information transfer of nodes and edges is carried out in an encoder composed of a fully  connected graph structure, and the encoder outputs the learned graph structure. Third, the  information transfer of nodes and edges is carried out in the graph structure learned by the  encoder, and the multi-position node features after information interaction are merged  through the attention mechanism to amplify the effective node feature information and  suppress invalid nodes feature information. Finally, the global sensor feature information  fused by the attention mechanism is used for activity recognition. 

<img src="IMG\multi_sensor.png" style="zoom: 33%;" />

<img src="IMG\attention.png" style="zoom: 33%;" />

Through a series of experiments, Porn has achieved the most advanced performance in  both wearable activity recognition data sets. In addition, this paper also demonstrates in a  visual way that the Porn model has effectively learned the connections and relationships  between the position sensor nodes through attention weight analysis and hidden layer graph structure analysis.

## Requirements

- Python 3.6.12

- Pytorch 1.2.0

## Data Preparation

Scripts to preprocess data can be found in the Data folder. To generate the data for training, validation and test, please run:

```python
cd data
python pre_process.py
```

<img src="IMG\pre_porcess.png" style="zoom: 33%;" />

## Run Experiments

You can run the Porn model in default dataset and other experiments setting by:

```python
cd ..
python train_Spirit.py
```

If you want to change the dataset to Realdisp, you can:

```python
python train_Spirit.py --dataset Realdisp
```

Also, if you want to tune the learning rate, you can:

```python
python train_Spirit.py --lr 0.0005
```

<img src="IMG\attention_ex.png" style="zoom: 33%;" />

<img src="IMG\graph_ex.png" style="zoom: 33%;" />