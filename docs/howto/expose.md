# How to expose classifier model

The classifer model may be exposed as an API for classification in different modes. 

## Exposing model with Open Inference Protocol API

First, being a transformer-based model, it is possible to use the HuggingFace-compatible KServe deployment. Specifically, within the platform this may be achieved as follows.

1. Create a HuggingFace serving deployment operation. It is required to update the 'code_src' url with github username and personal access token in the code cell below


```python
func = project.new_function(
    name="serve", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="git+https://<username>:<personal_access_token>@github.com/tn-aixpa/document-classifier",     
    handler="src.serve:serve",
    init_function="init",
    requirements=["numpy<2", "pandas==2.1.4","transformer_engine==1.12.0", "transformer_engine_cu12==1.12.0", "transformers==4.46.3", "torch==2.5.1", "torchmetrics==1.6.0"]
)
```

2. Activate the deployment.

The amount of data may be significant so the default container space may be not enough. The operation expects a volume attached to the container under ``/files`` path. Create a Persistent Volume Clain first and attach it to the run as shown below

```python
serve_run = func.run(
    action="serve",
    volumes=[{ 
            "volume_type": "persistent_volume_claim", 
            "name": "volume-document-classifier", 
            "mount_path": "/files", 
            "spec": { "claim_name": "volume-document-classifier" }
        }]
)
```

Once the deployment is activated, the V2 Open Inference Protocol is exposed and the Open API specification is available under ``/docs`` path.

3. Test the operation.

To test the functionality of the API, it is possible to use the V2 API calls. The "text" file contain the input text to be classified. The 'k' parameter specify the number of
classification labels required. For e.g. the request below asks for single classification label for input text.

```python
inputs = {"text": 'famiglia wifi ', "k": 1}
serve_run.invoke(json={"inference_input": inputs}).text
```

The api response will return the ids of most probable taxonomy. For futher details, look in to the correspondence.csv file present inside src folder which provide mapping between the ids and related taxonomy.

```
{
    "results": [
        46
    ]
}
```
