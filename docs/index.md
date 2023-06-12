Francesco Piccoli (ID: 5848474)

Marcus Plesner (ID: 4932021)

Boriss Bērmans (ID: 4918673)

# Project report - Seminar Computer Vision by Deep Learning (CS4245)  
## Introduction
<p align="justify">
This blog post presents the results of our reproduction and analysis of the paper "Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data" by J. You, X. Li, M. Low, D. Lobell, S. Ermon, presented at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17). The work was conducted as the project for the CS4245 Seminar Computer Vision by Deep Learning course 2022/23 at TU Delft. 
</p>

<p align="justify">
In the paper, the authors tackle a pressing global issue - accurately predicting crop yields before harvest, which is vital for food security, especially in developing countries. They introduce a scalable, accurate, and inexpensive method that significantly improves upon traditional techniques, the newly proposed method utilizes only remote sensing data (i.e. multispectral satellite images). Contrarily to previous approaches that required the use of hand-crafted features (i.e. vegetation indexes), the authors leverage advanced representation learning to let the deep learning model learn the features by itself. Moreover, differently from previous methods, the approach utilized in the paper is particurarly inexpensive as it does not rely on variables related to crop growth (i.e. local weather and soil properties) to model crop yield, which are often imprecise or lacking in developing countries where reliable yield predictions are most needed, instead, only worldwide available and freely accessible remote sensing data are used.
In addition to that, the authors propose an innovative dimensionality reduction technique by representing the raw multispectral satellite images as histograms of color pixel counts (using a mean-field approximation to achieve tractability), which makes it possible to train even when training data is scarce. Deep learning models (i.e. 3D-CNNs and LSTMs) are then trained on these histograms to predict crop yields.
Finally, the authors utilize Gaussian Processes to model spatiotemporal dependencies between datapoints (i.e. common soil properties between close crops). To evaluate the model they predict country-level soybean yield in the United States. The model proposed outperforms traditional remote-sensing based methods by 30% in terms of Root Mean Squared Error (RMSE). With slighly better performance for the 3D CNN architecture compared to the LSTM model.
</p>

<p align="justify">
Our work focused on first reproducing the results obtained by the authors, using the codebase available at <a href="[https://www.example.com](https://github.com/gabrieltseng/pycrop-yield-prediction)">this</a> github repository, to validate the claims made in the paper. We then expanded on this by experimenting with a Gated Recurrent Unit (GRU) model, on the hypothesis that it could be particularly effective with limited training data. Alongside this, we tested an encoder-only Transformer architecture, given the remarkable performance shown by transformers in modeling long-term dependencies (also) through the self-attention mechanisms. Finally, we aimed to assess the model's transferability and robustness by applying it to a new geography: soybean production in Italy, to verify whether the model can be trained on a country where labeled data is abundant and then be used to predict crop yield in countries where crop yield data is more scarce.
</p>

## Implementation
### Results validation/reproduction
<p align="justify">
In our attempt to reproduce the results presented in the paper, the first step was collecting the training data, comprising multispectral US-based satellite images, which were retrieved from the Google Earth Engine API. These data, amassing to over 100GB of .tif files, included the images, the masks for distinguishing farmland areas, and other .tif files providing information on the temperature encapsulated in the satellite images.
</p>

<i>??? Should we discuss in more detail the steps before training (export, preprocess, engineer) and maybe also the satellite images which satellite, which info...???</i>

<p align="justify">
Considering the sheer volume of data, storing it locally was not a feasible option. Thus, we leveraged Google Colab and Google Drive's premium plan for storing and accessing these files. Utilizing these cloud services made it easy and convenient to collaborate as we could share folders, avoid local storage, and exploit the free GPU resources provided by Colab for training the models.
</p>
<p align="justify">
To adapt the codebase to run on Colab, we imported the 'cyp' and 'data' folders from the GitHub repository to our shared Drive, which we then accessed from Colab. CondaColab, a tool that allows easy installation of Conda on Colab notebooks, was utilized for managing and installing the necessary dependencies. We then authenticated and created a project on the Google Earth Engine platform. The code we implemented on Colab was essentially analogous to the run.py file in the repository, with the addition of the training functions for the models we implemented ourselves - GRU and Tranformer. 
Following these steps, we were successful in setting up the training environment and reproducing the results as reported in the paper.
</p>

### GRU Architecture

<p align="justify">
The implementation of the GRU model is similar to the existing implementation of the LSTM model, without a cell state. The three parts of the GRU model are the GRU cells themselves, the dropout, which is applied at every sequence step, and the dense network which outputs the final predictions.
</p>

<p align="justify">
The following describes the GRU architecture. Note that every timestep of the GRU cell sequence processing applies dropout individually. This mirrors the way the DropoutWrapper class is used in the original paper
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/gru_architecture.png"/><br>
    <p align="center">Figure 1: The GRU architecture</p>
</p>

<p align="justify">
The reason for testing a GRU based network is because it has the potential to outperform an LSTM based network in certain scenarios. This is because a GRU is a simpler architecture. A GRU has two gates, update and reset, which control the flow of information. Therefore GRUs can be computationally more efficient and easier to train. Therefore GRUs exhibit better performance on tasks that require modeling short-term dependencies. Due to having fewer parameters, GRUs can capture important patterns in data quicker, enabling faster learning and better generalization.
</p>

### Tranformer Architecture
<p align="justify">
For the implementation of the Transformer model, we chose an encoder-only architecture, given that our task is a many-to-one, namely a sequence-to-number (i.e. the regression task of predicting the annual crop yield from the sequences of histograms representing the satellite images over the months), rather than sequence-to-sequence.
</p>
<p align="justify">
Our initial approach was to manually construct it from scratch by implementing all the necessary layers (i.e. self-attention, normalization..) to make the model highly customizable. However, this proved to be challenging and time-consuming, thus we resorted to utilizing PyTorch's built-in classes, namely <i><b>TransformerEncoderLayer</b></i> and <i><b>TransformerEncoder</b></i>. 
</p>
<p align="justify">
<i><b>TransformerEncoderLayer</b></i> represents the fundamental building block of the Transformer model. It comprises of a self-attention mechanism, a normalization layer and a feed-forward network. These layers are then stacked together by <i><b>TransformerEncoder</b></i> to construct the overall encoder model.
</p>
<p align="justify">
Our encoder-only Transformer model is made of several components. First, an embedding layer transforms the input histograms into a higher dimensional space. Subsequently, positional encodings are added, to make the model aware of the sequence's order. We implemented both components manually. Next, the model applies self-attention mechanisms, enabling it to focus on different parts of the input sequence to make predictions. Following this, we implemented an attention pooling mechanism to aggregate the sequence into a single vector representation. We experimented with various pooling methods over the sequence dimension, such as average and max pooling. However, attention pooling outperformed the others by a significant margin. In brief, attention pooling works by calculating attention scores and using them to weight the contribution of each sequence element, enabling the model to focus on the most relevant features. The pooling step was necessary, as the downstream regression task expects a single fixed-size input (a vector), not a sequence. Therefore, we need some method of condensing or summarizing the sequence of vectors into a single vector. Finally, a Feed-Forward Neural Network (FFNN) is used to predict the crop yield.
</p>
<p align="justify">
The architecture is highly configurable, allowing us to easily adjust and test key parameters as for instance the number of attention heads and encoder layers. Through experimentation, we found that due to the limited complexity of the input data, a low dropout rate, fewer heads and encoder layers, and a small hidden size yielded the best performance. More on this in the "Result" section.
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/encoder-only-architecture.png"/><br>
    <p align="center">Figure 1: The encoder-only transformer architecture.</p>
</p>

  
### New dataset: Soybean in Italy
<p align="justify">
...
</p>
  
## Results

To compare the performance of the models we have plotted the RMSE of the models for each year. As in the paper, the results are averaged over two runs to account for the random initialization and ropout during training. Models are always trained on all previous years. The results demonstrate that Gaussian Processes improve the performance of the models, and decreases the variance of the results.

| Year | LSTM | LSTM + GP | 3d CNN | 3d CNN + GP | GRU | GRU + GP | Transformer | Transformer + GP |
|------|------|-----------|--------|-------------|-----|----------|-------------|------------------|
| 2009 | 5.18 |    6.37   |  6.07  |     5.56    |5.75 |   6.67   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2010 | 7.27 |    7.30   |  6.75  |     7.03    |7.45 |   6.10   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2011 | 6.82 |    6.72   |  6.77  |     6.40    |6.26 |   5.83   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2012 | 7.01 |    6.46   |  5.91  |     5.72    |5.72 |   5.46   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2013 | 5.91 |    5.83   |  6.41  |     6.00    |6.51 |   5.98   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2014 | 5.99 |    4.65   |  5.28  |     4.87    |5.86 |   5.84   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |
| 2015 | 6.14 |    5.13   |  6.18  |     5.36    |6.59 |   5.72   | xxxxxxxxxxx | xxxxxxxxxxxxxxxx |



## Discussion and Conclusion

## References
