# Document-Classifier
A generic LLM-based document classifier

#### AIxPA

-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA

Multiclass sequence classifier based on BERT base italian, fine-tuned on selected corpora from Municipalities and organizations. 
The classifier is trained to suggest one or more labels from the taxonomy needed to categorize documents. 
When given a title, a description and an objective the classifier can predict the appropriate label from the taxonomy in use. 
The model can be trained for further fine-tuning on new data. The product contains operations for

- preprocessing the data in order to prepare it for training
- perform model training and registering the model
- serving the model as an optimized vLLM-based sequence classification API (Open Inference Protocol)
- serving the model using a custom API.

## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Preprocess data for training](./docs/howto/process.md)
- [Train the classifier model](./docs/howto/train.md)
- [Expose the classifed model as a service](./docs/howto/expose.md)
- [Evaluate](./docs/howto/evaluate.md)


## License

[Apache License 2.0](./LICENSE)