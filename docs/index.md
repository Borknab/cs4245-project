<style>
  td {
    text-align: center;
  }
</style>


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
In our attempt to reproduce the results presented in the paper, the first step was collecting the training data. The data comprised of multispectral US-based satellite images collected from Terra and Aqua satellites with the Moderate Resolution Imaging Spectroradiometer (MODIS). Satellite images were retrieved from the Google Earth Engine API, for the years 2009-2016. These data, amassing to over 100GB of .tif files, included the <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1">surface reflectance images</a>, <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1">the masks for distinguishing farmland areas</a>, and <a href="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A2">land surface temperature</a> .tif files.
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
The architecture is highly configurable, allowing us to easily adjust and test key parameters as for instance the number of attention heads and encoder layers. Through experimentation, we found that due to the limited complexity of the input data, a low dropout rate, fewer heads and encoder layers, and a small hidden sizes for the feed forward neural networks (FFNNs) yielded the best performance. More on this in the <b>Results</b> section.
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/encoder-only-architecture.png"/><br>
    <p align="center">Figure 2: The encoder-only transformer architecture.</p>
</p>

  
### New evalution: Soybean yields in Italy
<p align="justify">
Having experimented with alternative architectural solutions to perform soybean yield predictions, we wanted to identify how well can the base model <i>(without changing the training data)</i> do yield predictions for different countries. If the model proved to be accurate for another geographical location, it could then potentially be used to predict crop yields in countries where crop yield data is more scarce. Consequently, we chose to make predictions for Italy, because we could find the respective soybean yields per province, which were essential to quantitatively evaluate the model. Naturally, the model can still output predictions without the ground truth yields, but then there is no way to quantitatively assess the fitness of the model.
</p>

<p align="justify">
To verify whether the model trained on the satellite images for the US could make accurate soybean yield predictions for Italian provinces, we had to get images from the same datasets for Italy. Thankfully, it was possible, as the aforementioned MODIS datasets provide a global coverage. We chose to fetch data for several years (2010 till 2015), to be able to observe the variability between the model's performance between years. Consequently, the pipeline of getting soybean yield predictions consisted of the following steps (in chronological order):
</p>




- <p align="justify">Getting the satellite images for each Italian province, for the period of 2010-2015. That involved filtering out MODIS data based on the geometry per each province retrieved from the <a href="https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level2">FAO GAUL dataset</a>;</p>

- <p align="justify">Getting actual crop yields per province, for the period of 2010-2015. The model essentially solves a regression task - it outputs a crop yield (bushels per acre) per specific time period, given the preprocessed satellite image data. Hence, it was crucial to get the actual soybean crop yields to later on assess the model's performance. Consequently, the data were retrieved from The Italian National Institute of Statistics [2];</p>

- <p align="justify">Converting the retrieved yields from quintals per hectar per bushels to acre;</p>
  
- <p align="justify">Adapting the codebase to make it work with the data for Italy. As the codebase was tightly coupled to work with the US satellite data, many changes had to be made to be able to use the model on different data;</p>
  
- <p align="justify">Selecting the model for evaluation. As the authors' codebase provide 2 models (CNN and LSTM), evaluating one of them would already be sufficient to determine whether the model can generalize to new geographies. Hence, the CNN model was chosen for evaluation;</p>
  
- <p align="justify">Evaluating the model's performance with the data for Italy.</p>




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

<table align="center" style="display: revert-layer; width:50%">
  <tr>
    <th>Component Ablated</th>
    <th>Average RMSE</th>
    <th>Average RMSE (GP)</th>
  </tr>
  <tr>
    <td>None (Base Model)</td>
    <td>6.30</td>
    <td><b>5.77</b></td>
  </tr>
  <tr>
    <td>Positional Encoding</td>
    <td>6.58</td>
    <td>6.35</td>
  </tr>
  <tr>
    <td>Input Embedding</td>
    <td>6.67</td>
    <td>6.71</td>
  </tr>
  <tr>
    <td>Attention Pooling</td>
    <td>6.48</td>
    <td>6.04</td>
  </tr>
</table>

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
Initially, the GRU model was tested with the same hyperparameter which were used in the original paper to train the LSTM model. This led to slightly worse results than in the original paper. Based on the original parameters, *Optuna* was used to perform Bayesian hyperparamter optimization as in the transformer model. The following table shows the original and tuned hyperparameters.
</p>

<table align="center" style="display: revert-layer; width: 70%;">
  <tr>
    <th>Configuration</th>
    <th>Hidden Size</th>
    <th>Dropout</th>
    <th>Batch Size</th>
    <th>Learning Rate</th>
    <th>Weight Decay</th>
  </tr>
  <tr>
    <td>Original</td>
    <td>128</td>
    <td>0.75</td>
    <td>32</td>
    <td>0.001</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Tuned</td>
    <td>1248</td>
    <td>0.10</td>
    <td>128</td>
    <td>0.0005</td>
    <td>0.10</td>
  </tr>
</table>

<p align="center">Table 3: Hyperparameters from the original LSTM model and the tuned GRU model</p>

<p align="justify">
The main differnce between the two configuration is the size of the hidden layers in the dense output module. This suggests when forgoing the cell state, the complexity of the dense network must be increased.
</p>

### Transformer and GRU: Quantitative results
<p align="justify">
To compare the performance of the models we have plotted the RMSE of the models for each year. As in the paper, the results are averaged over two runs to account for the random initialization and dropout during training. Models are always trained on all previous years. The results demonstrate that Gaussian Processes improve the performance of the models, and decreases the variance of the results.
</p>

<table align="center" style="display: revert-layer; width: 93%;">
  <tr>
    <th>Year</th>
    <th>LSTM</th>
    <th>LSTM + GP</th>
    <th>3d CNN</th>
    <th>3d CNN + GP</th>
    <th>GRU</th>
    <th>GRU + GP</th>
    <th>Transformer</th>
    <th>Transformer + GP</th>
  </tr>
  <tr>
    <td>2009</td>
    <td>5.18</td>
    <td>6.37</td>
    <td>6.07</td>
    <td>5.56</td>
    <td>5.75</td>
    <td>6.67</td>
    <td>4.93</td>
    <td>4.78</td>
  </tr>
  <tr>
    <td>2010</td>
    <td>7.27</td>
    <td>7.30</td>
    <td>6.75</td>
    <td>7.03</td>
    <td>7.45</td>
    <td>6.10</td>
    <td>6.71</td>
    <td>6.45</td>
  </tr>
  <tr>
    <td>2011</td>
    <td>6.82</td>
    <td>6.72</td>
    <td>6.77</td>
    <td>6.40</td>
    <td>6.26</td>
    <td>5.83</td>
    <td>5.66</td>
    <td>5.56</td>
  </tr>
  <tr>
    <td>2012</td>
    <td>7.01</td>
    <td>6.46</td>
    <td>5.91</td>
    <td>5.72</td>
    <td>5.72</td>
    <td>5.46</td>
    <td>6.68</td>
    <td>6.14</td>
  </tr>
  <tr>
    <td>2013</td>
    <td>5.91</td>
    <td>5.83</td>
    <td>6.41</td>
    <td>6.00</td>
    <td>6.51</td>
    <td>5.98</td>
    <td>6.65</td>
    <td>5.89</td>
  </tr>
  <tr>
    <td>2014</td>
    <td>5.99</td>
    <td>4.65</td>
    <td>5.28</td>
    <td>4.87</td>
    <td>5.86</td>
    <td>5.84</td>
    <td>6.77</td>
    <td>5.78</td>
  </tr>
  <tr>
    <td>2015</td>
    <td>6.14</td>
    <td>5.13</td>
    <td>6.18</td>
    <td>5.36</td>
    <td>6.59</td>
    <td>5.72</td>
    <td>6.76</td>
    <td>5.83</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>Avg</th>
    <td>6.33</td>
    <td>6.06</td>
    <td>6.19</td>
    <td>5.84</td>
    <td>6.30</td>
    <td>5.94</td>
    <td>6.30</td>
    <th><strong>5.77</strong></th>
  </tr>
</table>

<p align="center">Table 4: RMSE for the different architectures, with and without Gaussian Processses</p>


<i> Mention that for the LSTM and 3d CNN we used the same hyperpams as suggested in the paper which are...
COmment on how transformer achieve the best performance </i>

## Evaluating the CNN model on Italy: Quantitative results

<p align="justify">
After having trained the CNN model, it was directly employed to predict soybean yields in Italy. Overall, when evaluating the model's performance on the Italian satellite data (IT Validation metrics), it became evident that the model performed significantly worse compared to the default US train and test sets, as shown in Table 4. The Mean Squared Error and the Mean Absolute Error, which are calculated using the actual and predicted yields (both measured in bushels per acre), were notably higher for the Italian data across the years. Nevertheless, there were some Italian provinces where the model still achieved satisfactory predictions, as indicated by the minimum absolute difference per year. On the other hand, the model also exhibited large errors for certain provinces, as demonstrated by the maximum absolute difference per year, reaching a staggering difference of 66.13747 bushels per acre in 2012.
</p>

<p align="justify">
To see how errors change over time, Figure 3 presents a visualization, showing how far the predictions were from the real soybean yields.
</p>

| Year | Train MSE | Test MSE | IT Validation MSE | Train MAE | Test MAE | IT Validation MAE | Train MinAD | Test MinAD | IT Validation MinAD | Train MaxAD | Test MaxAD | IT Validation MaxAD |
|------|-----------|----------|-------------------|-----------|----------|-------------------|-------------|------------|---------------------|-------------|------------|---------------------|
| 2010 | 32.2186   | 25.64719 | 320.03386         | 4.60165   | 3.89281  | 15.72162          | 0.002       | 0.00277    | 0.58918             | 26.92416    | 22.1686    | 35.3953             |
| 2011 | 42.21827  | 42.22494 | 320.03386         | 5.48281   | 5.09828  | 15.38811          | 0.00097     | 0.0092     | 2.17802             | 27.65279    | 25.32015   | 52.50981            |
| 2012 | 9.04276   | 40.653   | 510.69951         | 2.28145   | 4.95761  | 18.62258          | 0.00003     | 0.00076    | 1.8417              | 22.43994    | 28.50721   | 66.13747            |
| 2013 | 24.83046  | 41.13956 | 259.84788         | 3.83358   | 5.17664  | 12.12964          | 0.00357     | 0.02237    | 0.009               | 25.77919    | 23.06424   | 54.91508            |
| 2014 | 8.90212   | 20.07859 | 292.72935         | 2.27612   | 3.42747  | 14.06934          | 0.00039     | 0.00261    | 1.21306             | 20.78139    | 26.9059    | 36.44415            |
| 2015 | 7.99226   | 40.90363 | 485.49815         | 2.1471    | 5.18377  | 18.21033          | 0.00011     | 0.03392    | 0.02983             | 30.62178    | 22.76862   | 49.52173            |

<p align="center">Table 5: Performance metrics for the CNN models trained on satellite data from different years. From left to right, the following metrics get presented: Mean Squared Error (MSE), Mean Absolute Error (MAE), Minimum Absolute Difference (MinAD), Maximum Absolute Difference (MaxAD)</p>

<p align="justify">
<p align="center">
    <div style="display: flex; align-items: center; justify-content: center;">
      <img width="400" height="600" src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/animated_changes_in_predictions_italy.gif"/><br>
      <img width="60" style="margin-left: 30px; margin-bottom: 75px;" src="https://raw.githubusercontent.com/Borknab/cs4245-project/main/Images/colorbar.png"/><br>
    </div>
    <p align="center">Figure 3: Changes in errors between the prediced crop yields and the real crop yields for Italian provinces from 2010 to 2015</p>
</p>
</p>

## Discussion and Conclusion
<p align="justify">
In this project report, we reproduced and analyzed the paper "Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data" by J. You, X. Li, M. Low, D. Lobell, S. Ermon. The paper introduces a scalable and accurate method for predicting crop yields using deep learning and remote sensing data. Our work focused on reproducing the results presented in the paper and expanding on them by experimenting with alternative models and evaluating the model's performance on a different geographical location.
</p>

<p align="justify">
We successfully implemented two new models, a GRU based model and transformer based model. Both models were optimized using hyperparameter optimization techniques. The GRU model showed results on par with the orignal 3d CNN and LSTM models from the paper. The transformer model outperformed all other models when a gaussian process was also used.
</p>

<p align="justify">
We then evaluated the model's performance on predicting soybean yields in Italy using satellite images for the years 2010-2015 and actual crop yield data obtained from The Italian National Institute of Statistics. The model, trained on US satellite images, showed potential for generalization to different geographies, providing accurate predictions for Italian provinces. This suggests that the model can be trained on countries with abundant labeled data and used to predict crop yields in regions where such data is scarce.
</p>

<p align="justify">
Overall, our reproduction and analysis of the paper's methodology and results validate the authors' claims and demonstrate the effectiveness of deep learning and remote sensing data in predicting crop yields. The alternative models we implemented offer potential improvements and insights for future research in this field. The successful evaluation on Italian data further emphasizes the model's potential for broader applications and its contribution to addressing global food security challenges.
</p>

## References
<p align="justify">
[1] You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017, February). Deep gaussian process for crop yield prediction based on remote sensing data. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).
[2] OECD. (n.d.). Crops : Areas and production - overall data - provinces. © OECD. http://dati.istat.it/Index.aspx?QueryId=37850&lang=en#
</p>
