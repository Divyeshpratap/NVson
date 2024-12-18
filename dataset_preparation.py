# dataset_preparation.py

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import sys
import csv

import pandas as pd
from collections import Counter
from spacy.tokens import DocBin, Doc
import spacy
from sklearn.model_selection import train_test_split
import re

# ----------------------- Configuration Parameters ----------------------- #

DEFAULT_POS_INPUT_DIR = Path('data/pos_tagged/')
DEFAULT_LOG_DIR = Path('logs/')

DEFAULT_START_NUM = 1
DEFAULT_END_NUM = 2

# Define the set of valid Penn Treebank tags plus custom tags
VALID_TAGS = {
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
    'MD', 'SN', 'NSN', 'PDT', 'POS', 'PRP', 'RV', 'MV', 'NN', 'NNP', 'NNPS', 'NNS',
    'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB',
    'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``',
    ',', '.', ':', ';', '`', "''", '-LRB-', '-RRB-', '$', '#'
    # Add any other standard punctuation tags as needed
}

# Mapping dictionary for non-standard tags
TAG_MAPPING = {
    'HYPH': '-',        # Map HYPH to hyphen
    'XX': 'FW',         # Map XX to Foreign Word
    'NFP': 'X',         # Map NFP to Other
    '::': ':',          # Map double colon to single colon
    # Add more mappings as needed
}

# ----------------------- Logging Setup ----------------------- #

def setup_logging(log_dir: Path, log_file: str = "dataset_preparation.log"):
    """
    Set up logging configuration.

    :param log_dir: Directory where the log file will be saved.
    :param log_file: Name of the log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# ----------------------- Utility Functions ----------------------- #

def load_pos_tagged_document_pickle(pickle_file: Path, logger: logging.Logger) -> list:
    """
    Load POS-tagged tokens from a pickle file.

    :param pickle_file: Path to the pickle file.
    :param logger: Logger instance.
    :return: List of POS-tagged tokens.
    """
    if isinstance(pickle_file, str):
        pickle_file = Path(pickle_file)

    logger.info(f'Loading POS-tagged data from: {pickle_file}')
    try:
        with pickle_file.open('rb') as f:
            pos_tags = pickle.load(f)
        logger.info(f"Loaded POS-tagged data from {pickle_file}")
        return pos_tags
    except Exception as e:
        logger.error(f"Failed to load POS-tagged document from {pickle_file}: {e}")
        raise

def load_gpt_tagged_data(pickle_file: Path, logger: logging.Logger) -> Dict[int, Tuple[str, str]]:
    """
    Load GPT-generated noun/ verb tags from a pickle file.

    :param pickle_file: Path to the GPT-tagged pickle file.
    :param logger: Logger instance.
    :return: Dictionary with word positions and GPT tags.
    """
    logger.info(f'Loading GPT-tagged data from: {pickle_file}')
    try:
        with pickle_file.open('rb') as f:
            gpt_tags = pickle.load(f)
        logger.info(f"Loaded GPT-tagged data from {pickle_file}")
        return gpt_tags
    except Exception as e:
        logger.error(f"Failed to load GPT-tagged document from {pickle_file}: {e}")
        raise

def pos_tags_to_dataframe(pos_tags: list) -> pd.DataFrame:
    """
    Convert POS-tagged tokens to a pandas DataFrame.

    :param pos_tags: List of POS-tagged tokens.
    :return: pandas DataFrame with 'word' and 'pos' columns.
    """
    return pd.DataFrame(pos_tags)

def merge_pos_tags(pos_tags: List[Dict[str, str]], gpt_tags: Dict[int, Tuple[str, str]], pipeline_type: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Merge SpaCy POS tags with GPT-generated noun or verb classifications.

    :param pos_tags: List of POS-tagged tokens from SpaCy.
    :param gpt_tags: Dictionary of GPT-generated noun/verb classifications.
    :param pipeline_type: Type of pipeline ('noun' or 'verb').
    :param logger: Logger instance.
    :return: Merged pandas DataFrame with 'word' and 'newPOS' columns.
    """
    logger.info("Merging SpaCy POS tags with GPT-generated classifications.")

    df = pos_tags_to_dataframe(pos_tags)
    df['index'] = df.index
    df = df[['index', 'word', 'pos']]

    gpt_tags_df = pd.DataFrame.from_dict(gpt_tags, orient='index', columns=['wordPickle', 'posPickle'])
    gpt_tags_df = gpt_tags_df.reset_index().rename(columns={'index': 'index'})
    gpt_tags_df['index'] = gpt_tags_df['index'].astype(int)

    # Merge the Two DataFrames
    merged_df = pd.merge(df, gpt_tags_df, on='index', how='left')

    # Handle Missing Values
    merged_df[['wordPickle', 'posPickle']] = merged_df[['wordPickle', 'posPickle']].fillna('')

    # Create 'newPOS' column based on 'posPickle' and 'pos'
    merged_df['newPOS'] = merged_df['posPickle'].where(merged_df['posPickle'] != '', merged_df['pos'])

    # Replace specific POS tags in 'newPOS' with 'NSN' if pipeline is 'noun'
    if pipeline_type == 'noun':
        merged_df['newPOS'] = merged_df['newPOS'].replace(['NN', 'NNS', 'NNP', 'NNPS'], 'NSN')

    # Select the required columns
    df_out = merged_df[['word', 'newPOS']]
    return df_out

def save_merged_pos_tags(merged_tags: pd.DataFrame, output_file: Path, logger: logging.Logger):
    """
    Save the merged POS-tagged tokens to a tab-separated text file.

    :param merged_tags: pandas DataFrame with merged POS-tagged tokens.
    :param output_file: Path to the output text file.
    :param logger: Logger instance.
    """
    try:
        merged_tags.to_csv(
            output_file,
            sep='\t',
            index=False,
            header=False,
            quoting=csv.QUOTE_NONE,
            escapechar='\\'
        )
        logger.info(f"Saved merged POS tags to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save merged POS tags to {output_file}: {e}")
        raise

def validate_and_map_tags(tags: List[str], logger: logging.Logger) -> List[str]:
    """
    Validate POS tags and map non-standard tags to standard ones.

    :param tags: List of POS tags.
    :param logger: Logger instance.
    :return: List of validated and mapped POS tags.
    """
    validated_tags = []
    for tag in tags:
        if tag in VALID_TAGS:
            validated_tags.append(tag)
        elif tag in TAG_MAPPING:
            mapped_tag = TAG_MAPPING[tag]
            validated_tags.append(mapped_tag)
            logger.debug(f"Mapped non-standard tag '{tag}' to '{mapped_tag}'.")
        else:
            # Log the invalid tag and map it to 'X'
            validated_tags.append('X')
            logger.warning(f"Encountered invalid tag '{tag}'. Mapped to 'X'.")
    return validated_tags

# ----------------------- Main Pipeline Function ----------------------- #

def main():
    # ----------------------- Argument Parsing ----------------------- #

    # Initial parser to parse only pipelineType
    initial_parser = argparse.ArgumentParser(description="Dataset Preparation Pipeline for Custom POS Tagger", add_help=False)
    initial_parser.add_argument(
        '--pipelineType',
        type=str,
        choices=['noun', 'verb'],
        required=True,
        help="Noun or Verb Pipeline Type. Must be 'noun' or 'verb'"
    )

    # Parse known args to get pipelineType
    args, remaining_argv = initial_parser.parse_known_args()

    # Set defaults based on pipelineType
    if args.pipelineType == 'noun':
        PICKLE_INPUT_DIR_DEFAULT = Path('data/gpt_tagged/noun/')
        OUTPUT_DIR_DEFAULT = Path('data/final_pos/noun')
    elif args.pipelineType == 'verb':
        PICKLE_INPUT_DIR_DEFAULT = Path('data/gpt_tagged/verb/')
        OUTPUT_DIR_DEFAULT = Path('data/final_pos/verb')

    # Full parser with all arguments
    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Dataset Preparation Pipeline for Custom POS Tagger"
    )

    parser.add_argument(
        '--start_num',
        type=int,
        default=DEFAULT_START_NUM,
        help=f"Starting file number. Default: {DEFAULT_START_NUM}"
    )
    parser.add_argument(
        '--end_num',
        type=int,
        default=DEFAULT_END_NUM,
        help=f"Ending file number. Default: {DEFAULT_END_NUM}"
    )
    parser.add_argument(
        '--pos_input_dir',
        type=Path,
        default=DEFAULT_POS_INPUT_DIR,
        help=f"Directory containing SpaCy POS-tagged pickle files. Default: {DEFAULT_POS_INPUT_DIR}"
    )
    parser.add_argument(
        '--pickle_input_dir',
        type=Path,
        default=PICKLE_INPUT_DIR_DEFAULT,
        help=f"Directory containing GPT-tagged pickle files. Default: {PICKLE_INPUT_DIR_DEFAULT}"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory to save final POS-tagged files. Default: {OUTPUT_DIR_DEFAULT}"
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory to save log files. Default: {DEFAULT_LOG_DIR}"
    )

    # Now parse all arguments
    args = parser.parse_args()

    # ----------------------- Ensure Subdirectory Exists ----------------------- #
    if args.pipelineType == 'noun':
        output_subdir = Path('data/final_pos/noun')
    elif args.pipelineType == 'verb':
        output_subdir = Path('data/final_pos/verb')

    # Ensure the output subdirectory exists
    output_subdir.mkdir(parents=True, exist_ok=True)

    # ----------------------- Setup Logger ----------------------- #
    logger = setup_logging(args.log_dir)
    logger.info("Starting Dataset Preparation Pipeline")

    # ----------------------- Validate Arguments ----------------------- #
    if args.start_num > args.end_num:
        logger.error("START_NUM cannot be greater than END_NUM.")
        sys.exit(1)

    if not args.pos_input_dir.exists():
        logger.error(f"POS input directory does not exist: {args.pos_input_dir}")
        sys.exit(1)

    if not args.pickle_input_dir.exists():
        logger.error(f"GPT input directory does not exist: {args.pickle_input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory is set to: {args.output_dir}")

    # ----------------------- Initialize spaCy ----------------------- #
    try:
        logger.info("Loading spaCy blank English model.")
        nlp = spacy.blank("en")
        logger.info("spaCy model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        sys.exit(1)

    # ----------------------- Processing Loop ----------------------- #
    try:
        for num in range(args.start_num, args.end_num + 1):
            logger.info(f"\nProcessing File Number: {num}")

            # Define file paths
            pos_pickle_file = args.pos_input_dir / f'document_{num}.pickle'
            if args.pipelineType == 'noun':
                gpt_pickle_file = args.pickle_input_dir / f'book_part_{num}_dict_noun_comb.pkl'
                output_text_file = args.output_dir / f'book_part_{num}_noun.txt'
            elif args.pipelineType == 'verb':
                gpt_pickle_file = args.pickle_input_dir / f'book_part_{num}_dict_verb_comb.pkl'
                output_text_file = args.output_dir / f'book_part_{num}_verb.txt'                

            # Check if input files exist
            if not pos_pickle_file.exists():
                logger.warning(f"POS-tagged file does not exist: {pos_pickle_file}. Skipping File Number: {num}")
                continue
            if not gpt_pickle_file.exists():
                logger.warning(f"GPT-tagged file does not exist: {gpt_pickle_file}. Skipping File Number: {num}")
                continue

            # Load POS-tagged data
            pos_tags = load_pos_tagged_document_pickle(pos_pickle_file, logger)

            # Load GPT-tagged data
            gpt_tags = load_gpt_tagged_data(gpt_pickle_file, logger)

            # Merge POS tags with GPT tags
            merged_tags = merge_pos_tags(pos_tags, gpt_tags, args.pipelineType, logger)

            # Save merged POS tags to text file
            save_merged_pos_tags(merged_tags, output_text_file, logger)

            logger.info(f"Successfully merged and saved POS tags for File Number: {num}")

    except Exception as e:
        logger.error(f"An error occurred during merging POS tags: {e}")
        sys.exit(1)

    # ----------------------- Dataset Validation and Preparation ----------------------- #
    try:
        logger.info("Starting dataset validation and preparation.")

        # Initialize DocBin for storing SpaCy Docs
        doc_bin = DocBin()

        # Counter for invalid tags
        invalid_tags_counter = Counter()

        # Iterate over merged POS-tagged files
        for num in range(args.start_num, args.end_num + 1):
            if args.pipelineType == 'noun':
                output_text_file = args.output_dir / f'book_part_{num}_noun.txt'
            elif args.pipelineType == 'verb':
                output_text_file = args.output_dir / f'book_part_{num}_verb.txt'

            if not output_text_file.exists():
                logger.warning(f"Merged POS-tagged file does not exist: {output_text_file}. Skipping.")
                continue

            logger.info(f"Processing merged POS-tagged file: {output_text_file}")

            words = []
            tags = []

            try:
                with output_text_file.open('r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, start=1):
                        if '\t' in line:
                            word, tag = line.strip().split('\t', 1)
                            word = word.strip()
                            tag = tag.strip()

                            # Validate and map tags
                            if tag in VALID_TAGS:
                                validated_tag = tag
                            elif tag in TAG_MAPPING:
                                validated_tag = TAG_MAPPING[tag]
                                logger.debug(f"Mapped non-standard tag '{tag}' to '{validated_tag}' for word '{word}' in line {line_num}.")
                            else:
                                validated_tag = 'X'
                                invalid_tags_counter[tag] += 1
                                logger.warning(f"Encountered invalid tag '{tag}' for word '{word}' in line {line_num}. Mapped to 'X'.")

                            words.append(word)
                            tags.append(validated_tag)
                        else:
                            logger.warning(f"Skipping malformed line {line_num} in {output_text_file}: {line.strip()}")

                if words and tags:
                    # Create a SpaCy Doc
                    doc = Doc(nlp.vocab, words=words)

                    # Assign POS tags
                    for token, tag in zip(doc, tags):
                        token.tag_ = tag
                        # Avoid assigning to token.pos_ to prevent inconsistencies with non-UD tags

                    doc_bin.add(doc)
                    logger.info(f"Added document from {output_text_file} to DocBin.")

            except Exception as e:
                logger.error(f"Failed to process file {output_text_file}: {e}")
                continue

        # Output invalid tags
        if invalid_tags_counter:
            logger.info("Encountered the following invalid or non-standard tags:")
            for tag, count in sorted(invalid_tags_counter.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{tag}: {count}")
        else:
            logger.info("No invalid tags encountered.")

        # Split the dataset into training and development sets
        all_docs = list(doc_bin.get_docs(nlp.vocab))
        logger.info(f"Total documents collected: {len(all_docs)}")

        if not all_docs:
            logger.error("No documents to process. Exiting.")
            sys.exit(1)

        train_docs, dev_docs = train_test_split(all_docs, test_size=0.1, random_state=42)
        logger.info(f"Training documents: {len(train_docs)}")
        logger.info(f"Development documents: {len(dev_docs)}")

        # Create separate DocBins for training and development
        train_bin = DocBin(docs=train_docs)
        dev_bin = DocBin(docs=dev_docs)

        # Save the DocBins to disk
        train_bin.to_disk(args.output_dir / "train.spacy")
        dev_bin.to_disk(args.output_dir / "dev.spacy")
        logger.info(f"Saved training data to {args.output_dir / 'train.spacy'}")
        logger.info(f"Saved development data to {args.output_dir / 'dev.spacy'}")

    except Exception as e:
        logger.error(f"An error occurred during dataset preparation: {e}")
        sys.exit(1)

    logger.info("Dataset Preparation Pipeline completed successfully.")

# ----------------------- Entry Point ----------------------- #

if __name__ == '__main__':
    main()
