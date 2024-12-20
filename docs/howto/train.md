# How to train classifier model

To train the model, it is possible to use ``train`` operation that performs fine-tuning of the corresponding language model
for the classification task. 

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "document-classifier" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Define the function

Register the ``train`` function in the project. It is required to update the 'code_src' url with github username and personal access token in the code cell below

```python
func = project.new_function(
    name="train", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="git+https://<username>:<personal_access_token>@github.com/tn-aixpa/document-classifier", 
    handler="src.train:train",
    requirements=["accelerate==1.1.1", "datasets==3.1.0", "torch==2.5.1", "torch_tensorrt==2.5.0", "torchmetrics==1.6.0", "torchtext==0.18.0", "transformer_engine==1.12.0", "transformer_engine_cu12==1.12.0", "transformers==4.46.3", "pandas==2.2.3", "numpy==2.1.3", "numpyencoder==0.3.0", "scikit-learn==1.5.2", "scipy==1.14.1", "GitPython==3.1.43", "attrs==24.2.0", "async-timeout==5.0.1", "aiosignal==1.3.1", "aiohappyeyeballs==2.4.4", "aiohttp==3.11.9", "Unidecode==1.3.8"]
)
```
The function represent a Python operation and may be invoked directly locally or on the cluster.

3. Run the preprocess function

Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("train_data_it")
```

Furthermore, the amount of data may be significant so the default container space may be not enough. The operation expects a volume
attached to the container under ``/files`` path. Create a Persistent Volume Clain first and attach it to the run as shown below.


```python
train_run = func.run(action="job", inputs={"train_data": artifact.key},
					parameters={"data_path": "/data", "model_save_path": "/model", "target_model_name": "document-classifier"},
					volumes=[{ 
					"volume_type": "persistent_volume_claim", 
					"name": "volume-document-classifier", 
					"mount_path": "/files", 
					"spec": { "claim_name": "volume-document-classifier" }}]
					)
```

Here the training targets Italian and the corresponding base model is selected. The resulting model will be registered as the project model under the name ``document-classifier``.
