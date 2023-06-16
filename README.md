<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group 20: Seminar in Computer Vision by Deep Learning (CS4245) Project
The project replicates and ablates on the results of the paper "Deep Gaussian Processes for Crop Yield Prediction Based on Remote Sensing Data" bt You et al. [[paper](https://cs.stanford.edu/~ermon/papers/cropyield_AAAI17.pdf)]

We replicate the original model's results first, then apply modifications by replacing the LSTM network with a Gated Recurrent Unit (GRU), additionally we replace the LSTM with an 'encoder only' transformer architecture. We conduct our tests using also Italian soybean yield data, additionally to the original study's U.S-based data. Our goal is to explore the model's performance and potential improvements in predicting crop yields using different architectures, and explore the transferability (and performance) of the model to different states.

We utilized the code available at [this](https://github.com/gabrieltseng/pycrop-yield-prediction) Github repo, in our repository only the source code is available (`cyp` folder), check the original codebase for all the additional folders and files. We tagged with `[CS4245]` all the changes we made to the original code.

To evaluate the model on Italian provinces, we had to make multiple other changes to the existing codebase (which are also tagged with `[CS4245]`). Additionally, the `italy_processing_plotting/` directory contains other files and scripts that were used to get the predictions:
- `convert_csv.py` is the script for converting the data from The Italian National Institute of Statistics to the format expected by the paper authors' codebase;
- `it_data.csv` contains the [data](http://dati.istat.it/Index.aspx?QueryId=37850&lang=en#j) exported from **The Italian National Institute of Statistics**. Namely, the file contains soybean yields per province, for the years 2010-2015;
- `it_yield_data.csv` contains the output from running `convert_csv.py`;
- `provinces_plot.py` is the script for generating Italy error map that can be seen in our [blog post](https://borknab.github.io/cs4245-project/).

For more information on the implementation and the results, check the full blog post at: https://borknab.github.io/cs4245-project/

## Authors

Marcus Plesner (ID: 4932021)

Francesco Piccoli (ID: 5848474)

Boriss Bērmans (ID: 4918673)

## User Manual

The easiest way to run the code is to use the colab notebook we created: https://colab.research.google.com/drive/1ikbf1cgCTYS2HfWi5xfhntpAcElbfNs1?usp=sharing

Make sure to import the `cyp` folder (available in this repo) and the `data` folder(from the original repo) in your google drive. 

