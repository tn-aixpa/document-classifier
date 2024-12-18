# Document Classifier

## Usage Scenario
The main purpose of the tool is to provide the possibility to classify a text of a document. The classifier relies on LLM implementations for different languages. In this product the italian language is used. 

The resulting model may be deployed as a service or used for batch processing. It can be used in different applications to assist with the automated annotation and classification of official documents or similar texts. Since the underlying implementation relies on [HuggingFace transformer library](https://huggingface.co/docs/transformers/en/index), the resulting model is fully compatible with the corresponding instruments, such as [KServe](https://kserve.github.io/website/latest/) and AIxPA platform.

## Implementation Aspects

In order to reconstruct the classifier model, the presented project provides the necessary routines for acquiring and processing the data, as well as for the model training.

### Data preparation 

Once the data is downloaded, it should be prepared for training using the ``preprocessing`` operation (see [how to reprocess data for training](./howto/process.md)). The preprocessing performs the text embedding and splits the data into train and test sets. The operation is configured with the following parameters:

- ``langs`` (``it``): languages of the documents
- ``years`` (``all``): years (or "all" if the whole dataset should be considered)
- ``add_title`` (false) whether to include the document title or ``title_only`` (false) only consider document titles
- ``add_mt_do`` (false) whether to consider all the label types or only TC
- ``seeds`` (110): seeds for training
- ``max_length`` (512): max token lenght 

### Model training

The training data is ready for use by the ``train`` operation (see [how to train the classifier model](./docs/howto/train.md) for details). The operation relies on a series of hyper parameters typical for this kind of models:

- ``langs`` (``it``): language
- ``seeds`` (110): seeds
- ``device`` (cpu) device (e.g., CPU or GPU-based - ``cuda:0``)
- ``epochs`` (100): number of training epochs
- ``batch_size`` (8): batch size of the dataset
- ``learning_rate`` (3e-5): learning rate
- ``max_grad_norm`` (5): Gradient clipping norm
- ``threshold`` (0.5): Threshold for the prediction confidence
- Whether to enable the custom loss (``custom_loss``, false) and to enable the weighted bcewithlogits loss (``weighted_loss``, false)
- ``fp16`` (false): Enable fp16 mixed precision training.
- ``eval_metric`` (``f1_micro``) Evaluation metric
- ``full_metrics`` (false): Whether to extract full metric set for model metadata
- Whether to save training reports (``save_class_report``, false) and how frequently (``class_report_step``, 1).

For the realistic datasets the GPU is required for training of the model. 

### Model serving
Once model is ready, it is possible to expose a service (API) on top of the model. The model may be exposed in a standard manner, using KServe vLLM-based serving environment. In this case the model is exposed using the V2 Open Inference Protocol. Alternatively the ``serve`` operation provides an implementation of python-based Serverless function for exposing a custom API. The two implementation provides different ways to manage the results. In particular, the custom one allows for specifying the max number of labels to return and provides and confidence score regarding each label. 

Also in case of model training GPU is required for the inference.