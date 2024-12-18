# How to evaluate EUR Lex classifier model

To evaluate the model, it is possible to use ``evaluate`` operation that performs perdiction on test data
for the classification task. 

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "eurovoc-classifier-project" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Define the function

Register the ``evaluate`` function in the project

```python
func = project.new_function(
    name="evaluate", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="git+https://github.com/tn-aixpa/eurovoc-classifier", 
    handler="src.evaluate:evaluate",
    requirements=["transformers==4.26.1", "scikit-learn==1.2.2", "scikit-multilearn==0.2.0", "numpy==1.23.4", "lsg-converter==0.0.5", "sentence-transformers==2.2.2", "fastapi==0.95.2", "uvicorn==0.22.0", "python-dotenv==1.0.0", "compress_fasttext==0.1.3", "scipy==1.10.0", "nltk==3.8.1", "gensim==4.3.0", "ufal.udpipe==1.3.0.1", "pyyaml==6.0", "stop-words==2018.7.23", "spacy==3.5.1", "PageRange==0.4"]
)
```
The function represent a Python operation and may be invoked directly locally or on the cluster.

3. Run the preprocess function

Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("train_data_it")
model = project.get_model("eurovoc-classifier-110")
```

Run the function


```python
func.run(action="job", inputs={"train_data": artifact.key, "model": model.key}, parameters={"seeds": "110", "lang": "it", "batch_size": 8, "threshold": 0.5, "parents": "none", "trust_remote": False, "data_path": "/data", "models_path": "/model"})
```

Here the prediction targets Italian and the corresponding base model is loaded. The model is reported with all metric set, the prediction generates as output the confusion matrix.

The results of prediction will be registered as the project artifact under the name ``evaluation``.