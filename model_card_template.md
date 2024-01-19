# Model Card

For additional information, see the Model Card paper: [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
The model used is a random forest classifier.

## Intended Use
This model utilizes census data to predict whether an individual's income is over $50,000/year or under $50,000/year.

## Training Data

_Include information about the source, size, and characteristics of the training data._

Data comes from the 1994 Census data. It was originally extracted by Barry Becker. It comes from this website: https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data

_Include information about the source, size, and characteristics of the evaluation data._




**************************************



## Metrics
The following metrics were used to evaluate the model's performance:

- **Precision:** 0.7278
- **Recall:** 0.6057
- **F1 (Fbeta):** 0.6611

These scores indicate areas for potential improvement in the model.

## Ethical Considerations
The outcomes of this model may not be fully representative of real-world occurrences. It is essential to exercise caution when interpreting predictions, and this model should not be used to establish cause-and-effect relationships based on salary predictions.

## Caveats and Recommendations
The dataset used for training and evaluation is outdated, and its characteristics may not align with the current environment. Future improvements may be achieved by updating the dataset and considering additional features.
