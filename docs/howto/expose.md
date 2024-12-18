# How to expose classifier model

The classifer model may be exposed as an API for classification in different modes. 

## Exposing model with Open Inference Protocol API

First, being a transformer-based model, it is possible to use the HuggingFace-compatible KServe deployment. Specifically, within the platform this may be achieved as follows.

1. Create a HuggingFace serving deployment operation.

```python
llm_function = project.new_function("eurovoc-classifier",
                                    kind="huggingfaceserve",
                                    model_name="model",
                                    path="s3://datalake/eurovoc-classifier-project/model/eurovoc_classifier-110/f8026820-2471-4497-97f5-8e6d49baac5f/")

```

2. Activate the deployment.
```python
llm_run = llm_function.run(action="serve", profile="1xa100")
```

Here the deployment is triggered on top of a specific resource profile (``1xa100`` in this example) that corresponds to GPU-based resources.

Once the deployment is activated, the V2 Open Inference Protocol is exposed and the Open API specification is available under ``/docs`` path.

3. Test the operation.

To test the functionality of the API, it is possible to use the V2 API calls. 

```python
model_name = "model"
json = {
    "inputs": [
        {
            "name": "input-0",
            "shape": [2],
            "datatype": "BYTES",
            "data": ["Test del documento legale", "Test del documento legale nell'ambito finanza"],
        }
    ]
}

llm_run.invoke(model_name=model_name, json=json).json()
```

## Exposing model with Custom API

To deploy more specific API that takes into account the types of labels, it is possible to use the ``serve`` operation defined in the project.
Specifically, the following steps should be performed.

1. Register the ``custom_service`` deployment operation in the project

```python
custom_func = project.new_function(
    name="custom_service", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="git+https://github.com/tn-aixpa/eurovoc-classifier", 
    handler="src.serve:serve",
    init_function="init",
    requirements=["transformers==4.26.1", "scikit-learn==1.2.2", "scikit-multilearn==0.2.0", "numpy==1.23.4", "lsg-converter==0.0.5", "sentence-transformers==2.2.2", "fastapi==0.95.2", "uvicorn==0.22.0", "python-dotenv==1.0.0", "compress_fasttext==0.1.3", "scipy==1.10.0", "nltk==3.8.1", "gensim==4.3.0", "ufal.udpipe==1.3.0.1", "pyyaml==6.0", "stop-words==2018.7.23", "spacy==3.5.1", "PageRange==0.4"]
)
```

The function represent a Python Serverless function that should be deployed on cluster.

2. Deploy the API on the cluster.

```python
llm_run = custom_func.run(action="serve", profile="1xa100")
```
The deployments targets the GPU-based profile, and internally it loads the model (default name is "eurovoc-classifier-110") for serving.

It is possible to customize the deployment through the environment variables passed to deployment:

- ``DEVICE`` (default ``cuda:0``): device to use (e.g., ``cpu`` or ``cuda:<index>``)
- ``MODEL_NAME``: name of the model to use
- ``PRED_TYPE`` (default ``label``): define what to return as a result (id or label)
- ``THRESHOLD``: threshold for the confidence to consider
- ``MODEL_LANGUAGE`` (default ``it``): language to use
- ``TOP_K``: to which number limit the resulting labels

3. Test the API

This API expects a POST call with body in the following form:

```python
json = {
    "text": "Qualche testo da classificare"
}
llm_run.invoke(method="POST", json=json).json()
```

3. Use Streamlit for testing

Once exposed, the service may be integrated with a simple demo [Streamlit](https://streamlit.io/) application. To do this, in your workspace
install the required dependencies (see the [src/app-requirements.txt](../../src/app-requirements.txt)) and run the application:

```
streamlit run src/app.py
```

The application expects the service endpoint (host:port format) as a ``SERVICE_URL`` environment variable. If not provided, the app will ask the endpoint to be provided y the user.

The application returns the list of tags corresponding to the provided text.