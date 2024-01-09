<div align="center"> <h1> Bird song classification </h1> </div>
<div align="center"><a>Jovana&nbsp;Gentić</a></div>
<br>
<br>

"Bird song classification" is an audio classification model. We used 1D Convolutional Neural Networks (CNN) to learn a multiclass classifier on dataset of 5422 bird songs, belonging to 5 classes. We used spectrograms as inputs to the model . This repo is written in Tensorflow.

<div align="center"><img src="./images/Figure_4.png.PNG"></div>
<div align="center"><img src="./images/Figure_5.png.PNG"></div>

# Installation #
To download dataset, go to https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set/data. Dataset folder should be called `dataset` and it should be placed in the same folder as `data.py`.

```
pip install -r requirements
```

# Model training #
### Running the code
To train the model, execute command:

```
python train.py
```
<div align="center"><img src="./images/tblogs.PNG"></div>

# Model Evaluation #
To evaluate the model, execute command:

```
python evaluate.py
```
<div align="center"><img src="./images/metrics.PNG"></div>
<div align="center"><img src="./images/confusionmatrix.PNG"></div>
