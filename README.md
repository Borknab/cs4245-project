<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group 20: Seminar in Computer Vision by Deep Learning (CS4245) Project
The project replicates and ablates on the results of the paper "Deep Gaussian Processes for Crop Yield Prediction Based on Remote Sensing Data" bt You et al. [[paper](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf)]

We replicate the original model's results first, then apply modifications by replacing the LSTM network with a Gated Recurrent Unit (GRU), additionally we replace the LSTM with an 'encoder only' transformer architecture. We conduct our tests using also Italian soybean yield data, additionally to the original study's U.S-based data. Our goal is to explore the model's performance and potential improvements in predicting crop yields using different architectures, and explore the transferability (and performance) of the model to different states.

We utilized the code available at [this](https://github.com/gabrieltseng/pycrop-yield-prediction) Github repo, in our repository only the source code is available (`cyp` folder), check the original codebase for all the additional folders and files. We tagged with `[CS4245]` all the changes we made to the original code.

For more information, check the full blog post at: https://borknab.github.io/cs4245-project/

## Authors

Marcus Plesner (ID: 4932021)

Francesco Piccoli (ID: 5848474)

Boriss Bērmans (ID: 4918673)

## User Manual

The easiest way to run the code is to use the colab notebook we created: https://colab.research.google.com/drive/1ikbf1cgCTYS2HfWi5xfhntpAcElbfNs1?usp=sharing

Make sure to import the `cyp` folder (available in this repo) and the `data` folder(from the original repo) in your google drive. 



## Results


