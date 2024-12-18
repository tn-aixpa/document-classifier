# How to prepare data for training

To prepare the training data, it is required to log the data available in src folder 'addestramento.gzip' in the project context 

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "document-classifier" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Log the artifact

```python
artifact = project.log_artifact(name="train_data_it",
                    kind="artifact",
                    source="./addestramento.gzip")
)
```
Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("train_data_it")
```

The resulting dataset will be registered as the project artifact in the datalake under the name ``train_data_it``.
