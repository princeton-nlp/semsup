# README for RCV1

### Setup

Please complete the instructions in the main [README](README.md) before continuing with the RCV1 instructions.
Tested with python version `3.8`.

``` bash
pip install -r requirements.txt
```

### Getting the data
RCV1 (version 2) data is not available for distribution.
The following are instructions to obtain the data.

1. Request access to it from [this link](https://trec.nist.gov/data/reuters/reuters.html).
1. Download and store the folder as `./rcv1`.
1. Run the following bash script to preprocess RCV1 files - `cd preprocessing; bash rcv1_preprocess.sh; cd ..`

### Running scripts
Scripts for different scenarios and baselines are provided in [run_rcv1](run_rcv1).

For example, to run the DeViSE model on scenario 1, run the following:

``` bash
cd run_rcv1/scenario_1
bash devise.sh
```

The following are the scenarios

1. Scenario 1 - Unseen descriptions
1. Scenario 2 - Unseen classes
1. Scenario 3 - Unseen superclasses