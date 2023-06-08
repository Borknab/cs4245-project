Francesco Piccoli (ID: 5848474)

Marcus Plesner (ID: 4932021)

Boriss Bērmans (ID: 4918673)

# Project report - Seminar Computer Vision by Deep Learning (CS4245)  
## Introduction
<p align="justify">
This blog post presents the results of our reproduction and analysis of the paper "Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data" by J. You, X. Li, M. Low, D. Lobell, S. Ermon, presented at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17). The work was conducted as the project for the CS4245 Seminar Computer Vision by Deep Learning course 2022/23 at TU Delft. 
</p>

<p align="justify">
In the paper, the authors tackle a pressing global issue - accurately predicting crop yields before harvest, which is vital for food security, especially in developing countries. They introduce a scalable, accurate, and inexpensive method that significantly improves upon traditional techniques utilizing only remote sensing data (i.e. multispectral satellite images). Contrarily to previous approaches that required the use of hand-crafted features, the authors leverage advanced representation learning to let the deep learning model learn the features by itself. Moreover, differently from previous methods, the approach utilized in the paper is particurarly inexpensive as it does not rely on variables related to crop growth (such as local weather and soil properties) to model crop yield, which are often imprecise or lacking in developing countries where reliable yield predictions are most needed, instead, only worldwide available and freely accessible remote sensing data are used.
In addition to that, the authors propose an innovative dimensionality reduction technique by representing the raw multispectral satellite images as histograms of color pixel counts (using a mean-field approximation to achieve tractability), which makes it possible to train even when training data is scarce. Deep learning model (i.e. 3D-CNNs and LSTMs) are then trained on these histograms to predict crop yields.
Finally, the authors utilize Gaussian Processes to model spatiotemporal dependencies between datapoints (i.e. common soil properties between close crops). To evaluate the model they predict country-level soybean yield in the United States. The model proposed outperforms traditional remote-sensing based methods by 30% in terms of Root Mean Squared Error (RMSE). With slighly better performance for the 3D CNN architecture compared to the LSTM model.
</p>

<p align="justify">
Our work focused on first reproducing the results obtained by the authors, using the codebase available at [this](https://github.com/gabrieltseng/pycrop-yield-prediction) github repository, to validate the claims made in the paper. We then expanded on this by experimenting with a Gated Recurrent Unit (GRU) model, on the hypothesis that it could be particularly effective with limited training data. Alongside this, we tested an encoder-only Transformer architecture, given the remarkable performance shown by transformers in modeling long-term dependencies (also) through the self-attention mechanisms. Finally, we aimed to assess the model's transferability and robustness by applying it to a new geography: soybean production in Italy, to verify whether the model can be trained on a country where labeled data is abundant and then be used to predict crop yield in countries where crop yield data is more scarce.
</p>

## Implementation

## Results

## Discussion and Conclusion

## References
