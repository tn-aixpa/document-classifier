# Document Classifier

## Usage Scenario
The main purpose of the tool is to provide the possibility to classify a text of a document. The classifier relies on LLM implementations for different languages. In this product the italian language is used. 

The resulting model may be deployed as a service or used for batch processing. It can be used in different applications to assist with the automated annotation and classification of official documents or similar texts. Since the underlying implementation relies on [HuggingFace transformer library](https://huggingface.co/docs/transformers/en/index), the resulting model is fully compatible with the corresponding instruments, such as [KServe](https://kserve.github.io/website/latest/) and AIxPA platform.

## Implementation Aspects

In order to reconstruct the classifier model, the presented project provides the necessary routines for acquiring and processing the data, as well as for the model training.

### Data logging 

The data used for training can be found inside 'src' directory 'addestramento.gzip'. The data should be logged in the context of project for training  (see [how to log data for training](./howto/process.md)).

### Model training

The training data is ready for use by the ``train`` operation (see [how to train the classifier model](./docs/howto/train.md) for details). The operation relies on a series of hyper parameters typical for this kind of models:

- ``epochs`` (100): number of training epochs
- ``batch_size`` (8): batch size of the dataset
- ``learning_rate`` (3e-5): learning rate
- ``max_grad_norm`` (5): Gradient clipping norm
- ``weight_decay`` (0.005): Decoupled weight decay to apply.
- ``lr_scheduler_type`` ('linear'): By default, it uses “linear”. Internally, this retrieves get_linear_schedule_with_warmup scheduler from Trainer. Scheduler types (linear,cosine,cosine_with_restarts,polynomial,constant). For more info see [HuggingFace transformer library](https://huggingface.co/docs/transformers/en/index)

The template project also demonstrate the usage of compute_metrics function which takes the predictions and labels over the whole evaluation dataset and computes the metrics from them. The metrics are written in the model output directory inside metrics.json file.

### Model serving
Once model is ready, it is possible to expose a service (API) on top of the model. The model here is exposed using ``serve`` operation that provides an implementation of python-based Serverless function for exposing a custom API. The API allows for specifying the max number of labels to return. 

