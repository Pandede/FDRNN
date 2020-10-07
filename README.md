# FDRNN
Implementing
```Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2016). Deep direct reinforcement learning for financial signal representation and trading. IEEE transactions on neural networks and learning systems, 28(3), 653-664.```

## Preparation
1. Prepare the index data as **CSV** file. The file must include column *CloseDiff*, which represents the index difference
    `CloseDiff[t] = Index[t] - Index[t-1]`. The CSV files must arrange as following directory structure:
    ```
   +-- Data
   |    +-- futures
   |    |   +-- future_2018-01-01.csv
   |    |   +-- future_2018-01-02.csv
   |    |   +-- ...
   ```
2. To reduce the training time, it is **strongly recommended** that computing the parameters of fuzzy representation before training. The vanilla index file can be transformed into fuzzy version via applying `FuzzyStreamer` in `handler.py`.
    ```python
   from handler import FuzzyStreamer
   #streamer = FuzzyStreamer(<window size>, <fuzzy degree>)
   streamer = FuzzyStreamer(lag, fuzzy_degree)
   # streamer.transform(<folder of original index files>, <folder of fuzzy index files>)
   streamer = streamer.transform('./Data/futures/train', './Data/fuzzy_futures/train')
   ```
3. Adjust the required parameters in `config.ini`
    ```ini
    [default]
    # Number of training epochs
    epochs = 1000
    # Save the model each n epochs
    save_per_epoch = 20
    # Transaction cost
    c = 0.05
    # Window size
    lag = 50
    # Data path
    data_src = ./Data
    # Log path
    log_src = ./Pickle
    
    [fddrl]
    fuzzy_degree = 3
    ```
   
## Run
Running FDRNN - The proposed method in the paper

`python main.py`

Running baseline DDRL - The proposed model without fuzzy representation

`python baseline_ddrl.py`

Running baseline DRL - The proposed method without fuzzy representation and autoencoder

`python baseline_drl.py`