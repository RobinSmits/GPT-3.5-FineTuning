# GPT-3.5 Finetuning on a Dutch language dataset

## Introduction



## Dataset

The dataset used in this experiment is the [DpgMedia2019: A Dutch News Dataset for Partisanship Detection](https://github.com/dpgmedia/partisan-news2019) dataset.

It contains various parts but the main part I use in this repository is a subset from the original 104K news articles. For each article there is a label 'partisan' stating whether the article is partisan or not (True/False). The amount of partisan / non-partisan articles is roughly balanced.

The dataset was created by the authors with the aim to contribute to for example creating a partisan news detector. In the python code used in the experiments the specific dataset files are downloaded automatically. 

Checkout the github and paper for more information about the dataset and how it whas constructed. See the References for the information.

NOTE !! As of January 2023 not all files are available anymore in the original github location of the dataset. If you require the original files than download them from the [Kaggle Dataset](https://www.kaggle.com/datasets/rsmits/dpgmedia2019).

The training and validation subsets as used in this repository are available in the 'data' folder.

## Training and Validation Data Subsets

In the notebook 'Prepare_Train_and_Validation_Datasets.ipynb' the full dataset will be split into smaller subsets that are used for training and validation.

These files are available in the 'data' folder in this repository.

The primary reasons to create smaller subsets of the data are that a GPT (or other LLM's) model only needs a few thousand samples to perform finetuning on them. Smaller datasets also means less tokens and a smaller bill for your credit card ;-)

The smaller subsets have the following sample amounts:
* Training: 3069 samples
* Validation: 1559 samples

## Transformer model fine-tuning and validation

TODO

## Open LLM model fine-tuning and validation

The PolyLM 1.7B Open LLM model was pretrained on a multi-lingual dataset which also contained the Dutch language.

In the notebook 'Open_LLM_Training_And_Validation.ipynb' the PolyLM model is quantized to 4-bits and fine-tuned with a QLoRA setup to minimize the GPU memory footprint.

It achieves an accuracy of 83.4% on the validation set.

## GPT-3.5 fine-tuning and validation

The code for fine-tuning GPT-3.5 can be found in the notebook 'Finetune_GPT-3.5.ipynb'. Based on the training and validation CSV files specific files are created  and uploaded to OpenAI that are used for fine-tuning the GPT-3.5 model. The news articles are wrapped into a specific prompt that's engineered for the classification we would like the model to learn through fine-tuning.

The validation code is available in the notebook 'Validate_GPT-3.5.ipynb'. for each record in the validation set the text of the news article is wrapped in the prompt and OpenAI is called through the API to get the response from the chatcompletion.
The response is converted to a binary label and with the ground truth labels the final classification report is generated.

The fine-tuned OpenAI GPT-3.5 model achieves an accuraccy on the validation set of 90.8%.

Note that I used OpenAI for fine-tuning and validation and not Azure OpenAI.

## Model Comparison

Find below the achieved accuracy scores on the validation set for the 3 Transformer models and the GPT-3.5 model.

The GPT-3.5 model achieves a high accuracy score after fine-tuning.

The performance of the 3 Transformer models lag a little bit behind. They would clearly benefit from training on more data samples.

The Open LLM PolyLM achieves the lowest score.

| (LLM) Model Type | Validation Accuracy (%) Score |
|:---------------|----------------:|
| PolyLM 1.7B | 83.4 |
| Multi-lingual DistilBert | 86.0 |
| Multi-lingual Bert | 87.6 |
| Multi-linqual DeBERTa V3 | 88.3 |
| GPT-3.5 Turbo 0613 (fine-tuned) | 90.8 |

## Future Work

In the near future I will expand this repository with the following code, results and further analysis:
* DONE: Add 1 or 2 more regular multi-lingual Transformer models.
* Train 1 of the regular multi-lingual Transformer models on all (104K) available news articles.
* DONE: Add finetuning and validation for any Open LLM's that are pretrained on the Dutch language.
* Perform validation based on in-context learning with GPT-3.5
* Add finetuning and validation for GPT-4
* Perform all above steps for an additional Dutch dataset
* ??? Any further requests/ideas are welcome ... post your question or idea through an Issue.

## References

```
@misc{1908.02322,
  Author = {Chia-Lun Yeh and Babak Loni and Mariëlle Hendriks and Henrike Reinhardt and Anne Schuth},
  Title = {DpgMedia2019: A Dutch News Dataset for Partisanship Detection},
  Year = {2019},
  Eprint = {arXiv:1908.02322},
}
```