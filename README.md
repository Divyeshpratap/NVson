# NV-son: Noun Verb Syntax Observational Network

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Generative AI](https://img.shields.io/badge/Generative%20AI-green)
![LangChain](https://img.shields.io/badge/LangChain-blue)
![Llama](https://img.shields.io/badge/Llama-grey)
![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)
![spaCy](https://img.shields.io/badge/spaCy-000000?logo=spacy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-343541?logo=OpenAI&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-grey)



## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Technical Overview](#technical-overview)
  - [Pipeline Steps](#pipeline-steps)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

**NV-son: Noun Verb Syntax Observation Network** is an automated data annotation pipeline, and pos tagger model designed to enhance Part-of-Speech (POS) tagging for previously uncategorized tags. By integrating traditional NLP tools like SpaCy with modern architectures such as Large Language Models (LLMs), this project provides a unified end-to-end pipeline for POS tagging, which can be extended further to Named Entity Recognition (NER).
NLP tasks like POS tagging and NER require extensive resources for data annotation and preprocessing (ETL). This project targets these areas by creating an automated pipeline to preprocess data and leverage GPT/LLMs for annotating data into previously undefined categories. By simply changing the context provided to the LLM model, the pipeline can annotate data based on different classification schemes.
In the current application, the model is guided to classify nouns into Shape Nouns (SN) and Non-Shape Nouns (NSN) as per instructions from [Samuelson and Smith 1998](https://cogdev.sitehost.iu.edu/labwork/sam.pdf) and MacArthur-Bates Communicative Development Inventories (MB-CDIs). These definitions can be modified to instruct the LLM to classify words into other categories as needed.

## Architecture

- **Data preprocessing** The dataset used to train the model is scraped from [Project Gutenberg](https://www.gutenberg.org/) specifically from the short stories for kids section. The dataset is cleaned and stored in a single-line format to facilitate POS tagging by the GPT/LLM model. The overall dataset comprises 4 million words and 100k sentences.
- **GPT-Enhanced Noun Classification**:  Single-line sentences are passed to GPT to classify nouns into Shape Nouns (SN) and Non-Shape Nouns (NSN). This step eliminates the need for manual annotation by human annotators, reducing bias and improving annotation quality by leveraging the consistent expertise of the LLM.
- **Training**: The training process is flexible and can be configured using SpaCy's configuration file. This flexibility allows users to choose between transformer-based models and conventional RNN models based on available computational resources.


<div align="center">
  <img src="https://github.com/user-attachments/assets/57fa68b8-93ce-4a61-ab41-3eebff419120" alt="Training Quantitative Measures" width="75%">
  <p><i>Training Data Preparation and Annotation Pipeline</i></p>
</div>


### Pipeline Steps

1. **ETL Pipeline (`etl_pipeline.py`)**
   - **Extraction**: Load raw text data.
   - **Transformation**:
     - Filter lines based on word count.
     - Replace specific words using a predefined dictionary.
     - Convert text to lowercase.
     - Split text into manageable documents.
   - **Loading**:
     - Perform initial POS tagging using SpaCy's transformer model.
     - Save POS-tagged data in both `.txt` and `.pickle` formats.
     - Remove existing POS tags and create dictionaries.
     - Split dictionaries based on sentence boundaries and key count constraints.
     - Save processed dictionaries for GPT tagging.

2. **GPT Noun Classifier (`gpt_noun_classifier.py`)**
   - **Loading Data**: Load split dictionaries from the ETL pipeline.
   - **Classification**:
     - Use GPT/LLMs to classify nouns into Shape Nouns (SN) and Non-Shape Nouns (NSN).
     - Leverage detailed prompts and context descriptions to guide the model.
   - **Saving Output**: Store GPT-generated classifications in both `.pickle` and `.txt` formats.

3. **Dataset Preparation (`dataset_preparation.py`)**
   - **Merging Tags**: Combine SpaCy's POS tags with GPT-generated classifications.
   - **Validation**:
     - Ensure all POS tags are valid per Penn Treebank standards.
     - Map or handle non-standard and invalid tags.
   - **Dataset Creation**:
     - Convert merged data into SpaCy `Doc` objects.
     - Split the dataset into training and development sets.
     - Save the final datasets in SpaCy's `.spacy` binary format.
       
4. **Training**
   - **Flexible Training**: Leverage SpaCy's SOTA model training capabilities by choosing from varied model architecture like RoBERTa or single direction RNN. 


**Research Insights:**

Studies have demonstrated that GPT and similar LLMs can outperform human annotators in certain classification tasks due to their vast training data and ability to recognize intricate patterns. For instance, the [QLoRA](https://arxiv.org/abs/2305.14314) paper highlights how parameter-efficient fine-tuning methods enhance the performance of language models in specialized tasks, including data annotation and classification.

## Installation

1. **Clone the Repository**

   ```bash
   git clone git@github.com:Divyeshpratap/NVson.git

## Usage

1. **Run the ETL Pipeline**
This script handles data extraction, preprocessing, initial POS tagging, and dictionary creation.
    ```bash
    python etl_pipeline.py \
        --input_file_path data/raw_texts/stories.txt \
        --max_line_words 150 \
        --max_document_words 2500 \
        --output_pos_tagged_dir data/pos_tagged/ \
        --output_dict_dir data/dictionaries/ \
        --output_separated_dir data/separated/ \
        --min_keys_per_split 25 \
        --max_keys_per_split 70 \
        --log_dir logs/

2. **Run the GPT Noun Classifier**
This script leverages GPT/LLMs to classify nouns based on the processed data from the ETL pipeline.
      ```bash
      python noun_classifier.py \
          --start_book 1 \
          --end_book 3 \
          --input_dir data/separated/ \
          --output_dir data/gpt_tagged/ \
          --log_dir logs/

3. **Run the Dataset Preparation**
This script merges POS tags, validates them, and prepares the final dataset for model training.
    ```bash
    python dataset_preparation.py \
        --start_num 1 \
        --end_num 2 \
        --pos_input_dir data/pos_tagged/ \
        --pickle_input_dir data/gpt_tagged/ \
        --output_dir data/final_pos/ \
        --log_dir logs/

4. **Train using Spacy Tagger Model**

Once the final dataset is prepared, train the custom SpaCy POS tagger using the generated .spacy files.
    ```bash
    python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 1


## Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/2b8ece9e-3fb0-497b-98d5-2a2a41bdf5ae" alt="Training Quantitative Measures" width="50%">
  <p><i>Training Quantitative Measures</i></p>
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/03e7b274-338a-4d99-bb72-9bb89ee892a4" alt="Sample Sentence Tag" width="75%">
  <p><i>Generated Shape Noun tags for a sentence</i></p>
</div>

In the above results SN represents Shape Noun where NSN represents Non-Shape Nouns. The classification of train, pen, and apple into shape nouns is accurate as shape nouns are nouns that are primarily identified by their ditinct shape. However painting is marked as Non-shape nouns which is accurate as they can have various shape, and also cheese is marked correctly as NSN because it is primarily a material noun, and also is deformable.

## Contact
For more information, contact [Divyesh Pratap Singh](https://www.linkedin.com/in/divyesh-pratap-singh/)

## License

This project is licensed under the [MIT License](LICENSE).
