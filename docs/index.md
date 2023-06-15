Francesco Piccoli (ID: 5848474)

Marcus Plesner (ID: 4932021)

Boriss Bērmans (ID: 4918673)

# Project report - Seminar Computer Vision by Deep Learning (CS4245)  
## Introduction
<p align="justify">
This blog post presents the results of our reproduction and analysis of the paper "Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data" by J. You, X. Li, M. Low, D. Lobell, S. Ermon, presented at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17). The work was conducted as the project for the CS4245 Seminar Computer Vision by Deep Learning course 2022/23 at TU Delft. 
</p>

<p align="justify">
In the paper, the authors tackle a pressing global issue - accurately predicting crop yields before harvest, which is vital for food security, especially in developing countries. They introduce a scalable, accurate, and inexpensive method that significantly improves upon traditional techniques. Contrarily to previous approaches that required the use of hand-crafted features (i.e. vegetation indexes), the authors leverage advanced representation learning to let the deep learning model learn the features by itself. Moreover, differently from previous methods, the approach utilized in the paper is particurarly inexpensive as it does not rely on variables related to crop growth (i.e. local weather and soil properties) to model crop yield, which are often imprecise or lacking in developing countries where reliable yield predictions are most needed. Instead, only worldwide available and freely accessible remote sensing data (i.e. multispectral satellite images) are used.
In addition to that, the authors propose an innovative dimensionality reduction technique by representing the raw multispectral satellite images as histograms of color pixel counts (using a mean-field approximation to achieve tractability), which makes it possible to train even when training data is scarce. Deep learning model (i.e. 3D-CNNs and LSTMs) are then trained on these histograms to predict crop yields.
Finally, the authors utilize Gaussian Processes to model spatiotemporal dependencies between datapoints (i.e. common soil properties between close crops). To evaluate the model they predict country-level soybean yield in the United States. The proposed model outperforms traditional remote-sensing based methods by 30% in terms of Root Mean Squared Error (RMSE), with slighly better performance for the 3D CNN architecture compared to the LSTM model.
</p>

<p align="justify">
Our work focused on first reproducing the results obtained by the authors, using the codebase available at <a href="[https://www.example.com](https://github.com/gabrieltseng/pycrop-yield-prediction)">this</a> github repository, to validate the claims made in the paper. We then expanded on this by experimenting with a Gated Recurrent Unit (GRU) model, on the hypothesis that it could be particularly effective with limited training data. Alongside this, we tested an encoder-only Transformer architecture, given the remarkable performance shown by transformers in modeling long-term dependencies (also) through the self-attention mechanisms. Finally, as the model gets trained on the satellite images for United States by default, it was of our interest to identify whether this pre-trained model can generalize to completely new geographies. This could allow us to verify whether the model can be trained on a country where labeled data is abundant (e.g., US) and then be used to predict crop yield in countries where crop yield data is more scarce. For this task, we have chosen to verify the model on the satellite images for Italy.
</p>

## Implementation
### Results validation/reproduction
<p align="justify">
In our attempt to reproduce the results presented in the paper, the first step was collecting the training data. The data comprised of multispectral US-based satellite images collected from Terra and Aqua satellites with the Moderate Resolution Imaging Spectroradiometer (MODIS). Satellite images were retrieved from the Google Earth Engine API, for the years 2003-2016. These data, amassing to over 100GB of .tif files, included the <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1">surface reflectance images</a>, <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1">the masks for distinguishing farmland areas</a>, and <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A2">land surface temperature</a> .tif files.
</p>

<p align="justify">
Considering the sheer volume of data, storing it locally was not a feasible option. Additionally, as the authors' codebase makes use of Earth Engine API, storing locally was not a possible, since only Google Cloud Storage, Google Drive, or Earth Engine Storage were supported for exports https://developers.google.com/earth-engine/guides/exporting. Thus, we leveraged Google Colab and Google Drive's premium plan for storing and accessing these files. Utilizing Google Drive made it easy and convenient to collaborate as we could share folders, avoid local storage, and exploit the free GPU resources provided by Colab for training the models.
</p>

<p align="justify">
To adapt the codebase to run on Colab, we imported the 'cyp' and 'data' folders from the paper authors' GitHub repository to our shared Drive, which we then accessed from Colab. CondaColab, a tool that allows easy installation of Conda on Colab notebooks, was utilized for managing and installing the necessary dependencies. We then authenticated and created a project on the Google Earth Engine platform. The code we implemented on Colab was essentially analogous to the run.py file in the repository, with the addition of the training functions for the models we implemented ourselves - GRU and Tranformer, as well as utility funcions to facilitate exports and processing for Italian satellite images. Following these steps, we were successful in setting up the training environment and reproducing the results as reported in the paper.
</p>

### GRU Architecture

<p align="justify">
The implementation of the GRU model is similar to the existing implementation of the LSTM model, without a cell state. The three parts of the GRU model are the GRU cells themselves, the dropout, which is applied at every sequence step, and the dense network which outputs the final predictions.
</p>

<p align="justify">
The following describes the GRU architecture. Note that every timestep of the GRU cell sequence processing applies dropout individually. This mirrors the way the DropoutWrapper class is used in the original paper
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/gru_architecture.png" width="500px"/><br>
    <p align="center">Figure 1: The GRU architecture</p>
</p>

<p align="justify">
The reason for testing a GRU based network is because it has the potential to outperform an LSTM based network in certain scenarios. This is because a GRU is a simpler architecture. A GRU has two gates, update and reset, which control the flow of information. Therefore, GRUs can be computationally more efficient and easier to train Consequently, GRUs exhibit better performance on tasks that require modeling short-term dependencies. Due to having fewer parameters, GRUs can capture important patterns in data quicker, enabling faster learning and better generalization.
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
Our encoder-only Transformer model is made of several components. First, an embedding layer transforms the input histograms into a new embedded dimensional space. Subsequently, positional encodings are added, to make the model aware of the sequence's order. We implemented both components manually. Next, the model applies self-attention mechanisms, enabling it to focus on different parts of the input sequence to make predictions. Following this, we implemented an attention pooling mechanism to aggregate the sequence into a single vector representation. We experimented with various pooling methods over the sequence dimension, such as average and max pooling. However, attention pooling outperformed the others by a significant margin. In brief, attention pooling works by calculating attention scores and using them to weight the contribution of each sequence element, enabling the model to focus on the most relevant features. The pooling step was necessary, as the downstream regression task expects a single fixed-size input (a vector), not a sequence. Therefore, we need some method of condensing or summarizing the sequence of vectors into a single vector. Finally, a Feed-Forward Neural Network (FFNN) is used to predict the crop yield.
</p>

<p align="justify">
The architecture is highly configurable, allowing us to easily adjust and test key parameters as for instance the number of attention heads and encoder layers. Through experimentation, we found that due to the limited complexity of the input data, a low dropout rate, fewer heads and encoder layers, and a small hidden sizes for the feed forward neural networks (FFNNs) yielded the best performance. More on this in the "Result" section.
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/encoder-only-architecture.png"/><br>
    <p align="center">Figure 2: The encoder-only transformer architecture.</p>
</p>

  
### New evalution: Soybean yields in Italy
My text goes here

## Results

### Transformer and GRU: Hyperparamater optimizations
<p align="justify">
Once the models (Transformer and GRU) were implemented codewise, we tested them and we delved into hyperparameter optimization.
</p>

#### Tranformer 
<p align="justify">
For the Transformer, we started with a manual selection of values for embedding size, batch size, number of attention heads, hidden dimensions of the feedforward layer, dropout, and number of encoder layers, to find a subset of value ranges that showed promising results.
</p>

<p align="justify">
Taking the obtained initial results into consideration, we then employed the <i>Optuna</i> Python package to perform Bayesian hyperparameter optimization. The process returned an optimal configuration that revolved around lower values across the parameters - an embedding size between 48 to 128, attention heads in the range of 2 to 4, 1 to 3 encoder layers, and a low dropout rate from 0.1 to 0.2. A batch size of 64 demonstrated the best results. The best performing model was run with the following hyperparameters: 
</p>
    
| Num of Encoders | Embedding Size | Num of Attention Heads | Dropout Rate | FFNN (encoder) Hidden Size |   FFNN (last) Hidden Size  | Patience | Batch Size |
|-----------------|----------------|------------------------|--------------|----------------------------|----------------------------|----------|------------|
|        1        |       48       |           3            |     0.1      |            512             |            128             |    10    |     64     |

<p align="center">Table 1: Best hyperparamaters for the encoder-only transformer</p>   

<p align="justify">
The discovery that the optimal configuration leaned towards lower values for various parameters seem to indicate a relatively low complexity of the data domain. The model achieved optimal results without requiring a complex or deep architecture, which signifies that it was successful in feature extraction without resorting to overfitting (given also the low dropout rate).
</p>

<p align="justify">
An interesting observation from our experiments was the efficient training time of the Transformer. Despite its great performance, it trained in under 30 minutes, a significant difference from the <b>!!!N-hour!!!</b> training period required by the CNN and LSTM models. This showcases the exceptional efficiency of the Transformer architecture and paves the way for potential future research: with an expanded dataset, it is very likely that the Transformer's performance could outdo the other models by a substantial margin, whilst still maintaining a feasible training duration. This exploration of hyperparameters and model efficiency shows the power and potential of the Transformer architecture in our domain of application.
</p>

##### Ablation study 
<p align="justify">
To further validate the architectural choices and evaluate their individual contributions to the final performance of our model, we performed an ablation study. In the study, we systematically remove or replace certain parts of the model and measure the impact on performance. We ablated on positional encoding, input embedding and attention pooling, as they are essential components of our encoder-only Transformer architecture.
</p>

<p align="justify">
As shown in Table 2 below, the removal or replacement of any of these components led to a noticeable increase in RMSE values, which indicates a decrease in model performance. 
</p>

<center>
|       Component Ablated      | Average RMSE | Average RMSE (GP) |
|------------------------------|--------------|-------------------|
|       None (Base Model)      |     6.30     |    <b>5.77</b>    |
|      Positional Encoding     |     6.58     |       6.35        |
|       Input Embedding        |     6.67     |       6.71        |
|      Attention Pooling       |     6.48     |       6.04        |
</center>

<p align="center">Table 2: Ablation study results</p>

<p align="justify">
The ablation of positional encoding led to a noticeable performance degradation. Positional encoding in the Transformer model is crucial for understanding the temporal ordering in the sequence of satellite images, given that the Transformer architecture does not have inherent sequence awareness. Without positional encoding, the model struggled to effectively extract sequential patterns from the input data, resulting in a less accurate prediction.
</p>

<p align="justify">
When we removed input embedding and directly fed the input into the Transformer encoder (with input dimension = 288), we also observed a performance drop. The input embedding layer transforms the high-dimensional raw input into a lower-dimensional space where the Transformer can learn more effectively. Without this layer, the model was dealing directly with high-dimensional data, which likely made the training process more challenging and less effective.
</p>

<p align="justify">
Replacing attention pooling with average pooling resulted in another performance drop. The attention mechanism in the pooling layer allows the model to focus on more important aspects of the input sequence while predicting the crop yield. With average pooling, all elements in the sequence were treated with equal importance, leading to a loss of this valuable selective attention capability.
</p>

<p align="justify">
In conclusion, each of these ablations resulted in a degraded performance, indicating that each component – positional encoding, input embedding, and attention pooling – contributes significantly to the success of our model. The decrease in performance when removing any of these elements demonstrates their importance in the architecture and validates our initial architectural choices.
</p>

#### GRU 
<p align="justify">
hyperparams..
</p>

### Transformer and GRU: Quantitative results
<p align="justify">
To compare the performance of the models we have plotted the RMSE of the models for each year. As in the paper, the results are averaged over two runs to account for the random initialization and ropout during training. Models are always trained on all previous years. The results demonstrate that Gaussian Processes improve the performance of the models, and decreases the variance of the results.
</p>

<center>
|  Year  | LSTM | LSTM + GP | 3d CNN | 3d CNN + GP | GRU | GRU + GP | Transformer | Transformer + GP |
|--------|------|-----------|--------|-------------|-----|----------|-------------|------------------|
|  2009  | 5.18 |    6.37   |  6.07  |     5.56    |5.75 |   6.67   |     4.93    |       4.78       |
|  2010  | 7.27 |    7.30   |  6.75  |     7.03    |7.45 |   6.10   |     6.71    |       6.45       |
|  2011  | 6.82 |    6.72   |  6.77  |     6.40    |6.26 |   5.83   |     5.66    |       5.56       |
|  2012  | 7.01 |    6.46   |  5.91  |     5.72    |5.72 |   5.46   |     6.68    |       6.14       |
|  2013  | 5.91 |    5.83   |  6.41  |     6.00    |6.51 |   5.98   |     6.65    |       5.89       |
|  2014  | 5.99 |    4.65   |  5.28  |     4.87    |5.86 |   5.84   |     6.77    |       5.78       |
|  2015  | 6.14 |    5.13   |  6.18  |     5.36    |6.59 |   5.72   |     6.76    |       5.83       |
|        |      |           |        |             |     |          |             |                  |
|**Avg** | 6.33 |    6.06   |  6.19  |     5.84    |6.30 |   5.94   |     6.30    |     **5.77**     |
</center>

<p align="center">Table 2: RMSE for the different architectures, with and without Gaussian Processses</p>


<i> Mention that for the LSTM and 3d CNN we used the same hyperpams as suggested in the paper which are...
COmment on how transformer achieve the best performance </i>

## Evaluating the CNN model on Italy: Quantitative results

<p align="justify">
<p align="center">
    <img width="400" height="600" src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/animated_changes_in_predictions_italy.gif"/><br>
    <p align="center">Figure 3: Changes in errors between the prediced crop yields and the real crop yields for Italian provinces from 2010 to 2015</p>
</p>
</p>

<div align="center">
| Year | Train MSE | Test MSE | IT Validation MSE | Train MAE | Test MAE | IT Validation MAE | Train MinAD | Test MinAD | IT Validation MinAD | Train MaxAD | Test MaxAD | IT Validation MaxAD |
|------|-----------|----------|-------------------|-----------|----------|-------------------|-------------|------------|---------------------|-------------|------------|---------------------|
| 2010 | 32.2186   | 25.64719 | 320.03386         | 4.60165   | 3.89281  | 15.72162          | 0.002       | 0.00277    | 0.58918             | 26.92416    | 22.1686    | 35.3953             |
| 2011 | 42.21827  | 42.22494 | 320.03386         | 5.48281   | 5.09828  | 15.38811          | 0.00097     | 0.0092     | 2.17802             | 27.65279    | 25.32015   | 52.50981            |
| 2012 | 9.04276   | 40.653   | 510.69951         | 2.28145   | 4.95761  | 18.62258          | 0.00003     | 0.00076    | 1.8417              | 22.43994    | 28.50721   | 66.13747            |
| 2013 | 24.83046  | 41.13956 | 259.84788         | 3.83358   | 5.17664  | 12.12964          | 0.00357     | 0.02237    | 0.009               | 25.77919    | 23.06424   | 54.91508            |
| 2014 | 8.90212   | 20.07859 | 292.72935         | 2.27612   | 3.42747  | 14.06934          | 0.00039     | 0.00261    | 1.21306             | 20.78139    | 26.9059    | 36.44415            |
| 2015 | 7.99226   | 40.90363 | 485.49815         | 2.1471    | 5.18377  | 18.21033          | 0.00011     | 0.03392    | 0.02983             | 30.62178    | 22.76862   | 49.52173            |
</div align="center">

<p align="center">Table 3: Performance metrics for the CNN models trained on satellite data from different years. From left to right, the following metrics get presented: Mean Squared Error (MSE), Mean Absolute Error (MAE), Minimum Absolute Difference (MinAD), Maximum Absolute Difference (MaxAD)</p> 

## Discussion and Conclusion
<p align="justify">
...
</p>

## References
<p align="justify">
[1] You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017, February). Deep gaussian process for crop yield prediction based on remote sensing data. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).
</p>
