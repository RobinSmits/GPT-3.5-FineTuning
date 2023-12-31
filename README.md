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

The following regular Transformer models where trained and validated in the notebook 'Transformer_Model_Training_And_Validation.ipynb' on the smaller training and validation data subsets:
* Multi-lingual DistilBert
* Multi-lingual Bert
* Multi-linqual DeBERTa V3

## Open LLM model fine-tuning and validation

The PolyLM 1.7B and OpenLLaMA 7B V2 Open LLM models where both pretrained on multi-lingual datasets which also contained the Dutch language.

In the notebook 'Open_LLM_Training_And_Validation.ipynb' the Open LLM models are quantized to 4-bits and fine-tuned with a QLoRA setup to minimize the GPU memory footprint.

The LoraConfig is set with a rank of 64 and alpha of 16.

After training the PolyLM 1.7B model achieves an accuracy on the validation set of 84.4% while the OpenLLaMA 7B V2 model even achieves 89.5%.

I did multiple training runs and on various occassions both models scored up to 0.5% higher or lower compared with the above mentioned value.  

## GPT-3.5 fine-tuning and validation

The code for fine-tuning GPT-3.5 can be found in the notebook 'Finetune_GPT-3.5.ipynb'. Based on the training and validation CSV files specific files are created  and uploaded to OpenAI that are used for fine-tuning the GPT-3.5 model. The news articles are wrapped into a specific prompt that's engineered for the classification we would like the model to learn through fine-tuning.

The validation code is available in the notebook 'Validate_GPT-3.5.ipynb'. for each record in the validation set the text of the news article is wrapped in the prompt and OpenAI is called through the API to get the response from the chatcompletion.
The response is converted to a binary label and with the ground truth labels the final classification report is generated.

The fine-tuned OpenAI GPT-3.5 model achieves an accuraccy on the validation set of 89.4%. 

Note that I used OpenAI for fine-tuning and validation and not Azure OpenAI.

In the latest version (December 5th, 2023) of this notebook I have made the following updates:
* Updated to the latest version of OpenAI (1.3.7) and modified the API calls accordingly.
* Changed the model to the latest version: "gpt-3.5-turbo-1106".

## Model Comparison

Find below the achieved accuracy scores on the validation set for the 3 Transformer models and the GPT-3.5 model.

The GPT-3.5 model achieves a high accuracy score after fine-tuning.

The performance of the 3 Transformer models lag a little bit behind. They would clearly benefit from training on more data samples.

The Open LLM PolyLM achieves the lowest score. The OpenLLaMA 7B V2 model however achieves a remarkable score of 89.5% which is comparable to the scores achieved by the GPT-3.5 Turbo finetuned models (0613 and 1106).

| (LLM) Model Type | Validation Accuracy (%) Score |
|:---------------|----------------:|
| PolyLM 1.7B (Lora: r = 64) | 84.4 |
| Multi-lingual DistilBert | 85.6 |
| Multi-lingual Bert | 86.3 |
| Multi-linqual DeBERTa V3 | 85.8 |
| OpenLLaMA 7B V2 (Lora: r = 64) | 89.5 |
| GPT-3.5 Turbo 0613 (fine-tuned) | 90.8 |
| GPT-3.5 Turbo 1106 (fine-tuned) | 89.4 |
| GPT-3.5 Turbo 1106 (in-context learning) | 56.0 |
| !! Multi-linqual DeBERTa V3 (full dataset) | 95.2 |

## References

```
@misc{1908.02322,
  Author = {Chia-Lun Yeh and Babak Loni and Mariëlle Hendriks and Henrike Reinhardt and Anne Schuth},
  Title = {DpgMedia2019: A Dutch News Dataset for Partisanship Detection},
  Year = {2019},
  Eprint = {arXiv:1908.02322},
}
```