# How to prepare data for training

In this template project, it is demonstrated to train classification model with BERT (Bidirectional Encoder Representations from Transformers). The classifier model is trained to suggest one or more labels within the
input data. The training data looks like following table

| |text| labels |
|-|----|--------|
|0|text1....| 11
|1|text2....| 12

Prepare a balanced dataset of your choice. The template project works fine if you use your own balanced dataset which contain labels and text. For an example, see  [sample train dataset](/src/train_data.csv). 
The project performs all the necessary steps including label encoding, preparation of test, validate, and training dataset from the input dataset making it ready for model training.

For the sake of tutorial it is demonstrated below, how one can register the dataset inside to the digitalhub platform.

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "document-classifier" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Log the artifact

```python
URL='https://raw.githubusercontent.com/tn-aixpa/document-classifier/refs/heads/main/src/train_data.csv'
di = project.new_dataitem(name="train_data",kind="table",path=URL)
```
Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
di = project.get_dataitem("train_data")
di.key
```

The resulting dataset will be registered as the project dataitem in the datalake under the name ``train_data``.
