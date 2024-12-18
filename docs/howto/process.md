# How to prepare data for training

To prepare the training data, it is possible to use ``process`` operation that performs data scraping. 

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "eurovoc-classifier-project" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Define the function

Register the ``preprocess`` function in the project

```python
func = project.new_function(
    name="preprocess", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="git+https://github.com/tn-aixpa/eurovoc-classifier", 
    handler="src.preprocess:preprocess_data",
    requirements=["transformers==4.26.1", "scikit-learn==1.2.2", "scikit-multilearn==0.2.0", "numpy==1.23.4", "lsg-converter==0.0.5", "sentence-transformers==2.2.2", "fastapi==0.95.2", "uvicorn==0.22.0", "python-dotenv==1.0.0", "compress_fasttext==0.1.3", "scipy==1.10.0", "nltk==3.8.1", "gensim==4.3.0", "ufal.udpipe==1.3.0.1", "pyyaml==6.0", "stop-words==2018.7.23", "spacy==3.5.1", "PageRange==0.4"]
)
```
The function represent a Python operation and may be invoked directly locally or on the cluster.

3. Run the preprocess function

Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("classified_data")
```

Furthermore, the amount of data may be significant so the default container space may be not enough. The operation expects a volume
attached to the container under ``/files`` path. Create a Persistent Volume Clain first and attach it to the run as shown below.

```python
func.run(action="job", inputs={"classified_data": artifact.key}, 
         parameters={"years": "all", "langs": "it", "add_mt_do": True},
         volumes=[{ 
            "volume_type": "persistent_volume_claim", 
            "name": "volume-eurovoc", 
            "mount_path": "/files", 
            "spec": { "claim_name": "volume-eurovoc" }
        }])
```

Here the data for Italian language is used, considering all the available years and other parameters set to default. See [Usage](../usage.md) for further details.

The resulting dataset will be registered as the project artifact in the datalake under the name ``train_data_<lang>``.
