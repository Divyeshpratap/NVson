# NV-son: Noun Verb Syntax Observational Network

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Generative AI](https://img.shields.io/badge/Generative%20AI-green)
![LangChain](https://img.shields.io/badge/LangChain-blue)
![Llama](https://img.shields.io/badge/Llama-grey)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technical Overview](#technical-overview)
  - [Pipeline Steps](#pipeline-steps)
  - [Leveraging GPT/LLMs for Data Annotation](#leveraging-gptllms-for-data-annotation)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

**NVson Noun Verb Syntax Observation Network for GPT/ LLM Enhanced Automatic Data Annotation pipeline**. This project presents a comprehensive pipeline designed to streamline and enhance the process of Part-of-Speech (POS) tagging for natural language processing (NLP) tasks. By integrating traditional NLP tools with advanced Generative Pre-trained Transformers (GPT) and Large Language Models (LLMs), this pipeline one command pipeline to create a custom POS tagger from scratch.

## Features

- **Comprehensive ETL Process**: Extract, transform, and load raw text data with preprocessing steps like filtering, word replacement, and case normalization.
- **Advanced POS Tagging**: Utilize SpaCy's transformer-based models for initial POS tagging.
- **GPT-Enhanced Noun Classification**: Leverage GPT/LLMs to classify nouns into Shape Nouns (SN) and Non-Shape Nouns (NSN), surpassing traditional tagging methods.
- **Data Merging and Validation**: Combine SpaCy's POS tags with GPT-generated classifications, ensuring data integrity and consistency.
- **Automated Dataset Preparation**: Convert merged data into SpaCy's `.spacy` format, ready for model training with train-dev splits.
- **Flexible Configuration**: Parameterize key variables for customizable processing based on user requirements.
- **Robust Logging and Error Handling**: Maintain comprehensive logs for monitoring and debugging.

## Technical Overview

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

### Leveraging GPT/LLMs for Data Annotation

The integration of GPT and LLMs into the data annotation pipeline marks a significant advancement in NLP workflows. Traditional POS tagging relies on predefined rules and statistical models, which, while effective, have limitations in handling nuanced language patterns and context-specific classifications.

**Advantages of Using GPT/LLMs:**

- **Contextual Understanding**: GPT models excel in grasping the context, allowing for more accurate and context-aware annotations.
- **Adaptive Learning**: These models can adapt to specific classification schemes, such as distinguishing between Shape Nouns (SN) and Non-Shape Nouns (NSN), based on detailed instructions.
- **Efficiency**: Automating complex annotation tasks reduces manual effort and accelerates the data preparation process.

**Research Insights:**

Studies have demonstrated that GPT and similar LLMs can outperform human annotators in certain classification tasks due to their vast training data and ability to recognize intricate patterns. For instance, the [QLoRA](https://arxiv.org/abs/2305.14314) paper highlights how parameter-efficient fine-tuning methods enhance the performance of language models in specialized tasks, including data annotation and classification.

By incorporating GPT/LLMs into the POS tagging pipeline, this project leverages cutting-edge AI to achieve higher accuracy and consistency in data annotation, ultimately contributing to more robust and reliable NLP models.

## Installation

1. **Clone the Repository**

   ```bash
   git clone git@github.com:Divyeshpratap/NVson.git
