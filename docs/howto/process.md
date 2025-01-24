# How to prepare data for training

In this template project, it is demonstrated to train classification model with BERT (Bidirectional Encoder Representations from Transformers). The classifier model is trained to suggest one or more labels within the
input data. The training data looks like following table

| |text| labels |
|-|----|--------|
|0|text1....| 11
|1|text2....| 12

Prepare a balanced dataset of your choice. For an example, see  [sample train dataset](/src/train_data.csv). For the sake of tutorial it is demonstrated below, how one can register the dataset inside to the digitalhub platform.

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "document-classifier" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Log the artifact

```python
URL='https://raw.githubusercontent.com/tn-aixpa/document-classifier/refs/heads/main/src/train_data.csv'
artifact = project.log_dataitem(name="train_data",
                    kind="table",
                    source=URL)
```
Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("train_data")
artifact.key
```

The resulting dataset will be registered as the project artifact in the datalake under the name ``train_data``.
