# Document-Classifier
A generic LLM-based document classifier

#### AIxPA

-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA

A template project for Multiclass sequence classifier based on BERT base italian, fine-tuned on test dataset. The classifier is trained to suggest one or more labels from the taxonomy needed to categorize documents. When given a title, a description and an objective the classifier can predict the appropriate label from the taxonomy in use. The model can be trained for further fine-tuning on new data. The project demonstrated the necessary steps required to prepare dataset. The dataset labels are encoded and in the later stage it is split in to train, test, and validation sets in order to be ready for model training. The parameters of training can be configured as demonstrated. The project further demonstrate the approach to serve the trained model. The template project demonstrate instruction to

- Prepare and register train dataset
- Perform training and model regitration.
- Serve the generated model using a custom API.

## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Prepare and register dataset for training](./docs/howto/process.md)
- [Train the classifier model](./docs/howto/train.md)
- [Expose the classifed model as a service](./docs/howto/expose.md)


## License

[Apache License 2.0](./LICENSE)
