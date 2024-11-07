# gpt_verb_classifier.py

import os
import json
import pickle
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys
from dotenv import load_dotenv
load_dotenv() 
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# ----------------------- Logging Setup ----------------------- #

def setup_logging(log_dir: Path, log_file: str = "gpt_verb_classifier.log"):
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

# ----------------------- Configuration Parameters ----------------------- #

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_INPUT_DIR = Path('data/separated/')
DEFAULT_OUTPUT_DIR = Path('data/gpt_tagged/durativePunctual/')
DEFAULT_LOG_DIR = Path('logs/')

DEFAULT_START_BOOK = 1
DEFAULT_END_BOOK = 1

MODEL = 'gpt-4o-mini'  # Options: 'gpt-4o-mini', 'llama3.1:70b', 'llama2:latest'

# Tag Descriptions and Prompt Template
TAG_DESCRIPTIONS = """
**Verb Classification: Durative/Telicity (D/T)**
You are tasked with classifying action verbs in a sentence into four distinct categories based on their durativity and telicity:
- **Durative Telic (dute)**:
    - **Definition**: Verbs that denote actions extending over time with a clear endpoint.
    - **Examples**: build, complete, finish, run a marathon.
    - **Usage**: "She **built** a sandcastle."

- **Punctual Telic (pute)**:
    - **Definition**: Verbs that denote instantaneous actions with a clear endpoint.
    - **Examples**: hit, knock, sneeze, arrive.
    - **Usage**: "He **knocked** on the door."

- **Durative Atelic (duat)**:
    - **Definition**: Verbs that denote actions extending over time without a specific endpoint.
    - **Examples**: run, swim, read, talk.
    - **Usage**: "She **read** books every evening."

- **Punctual Atelic (puat)**:
    - **Definition**: Verbs that denote instantaneous actions without a specific endpoint.
    - **Examples**: blink, clap, nod, laugh.
    - **Usage**: "He **clapped** his hands."

**Key Guidelines**:

1. **Exclusive Classification**: Each action verb should be classified into one of the four categories based on its primary lexical meaning in the given context.
2. **Polysemy Handling**: Some verbs have multiple senses; classify them based on context.
    - **Example**:
        - **dute**: "She **runs** a successful business." → **dute**
        - **duat**: "She **runs** every morning." → **duat**
3. **Context-Dependent Classification**:
    - **"Cook"**:
        - **dute**: "He **cooked** dinner for the family." (Implying the completion of dinner)
        - **duat**: "He **cooked** all afternoon." (Focusing on the duration)
    - **"Play"**:
        - **duat**: "They **played** in the park." (Ongoing activity)
        - **pute**: "They **played** a game." (Specific event)
4. **Supplementary Classification Method**:
    - **"Three Times in One Minute"** Sentence Frame:
        - **Purpose**: Provides an additional method to determine verb classification.
        - **Guidelines**:
            - If adding "three times in one minute" makes the sentence acceptable, classify the verb as **Punctual** (puat or pute).
            - If it makes the sentence less acceptable or changes the meaning significantly, classify the verb as **Durative** (duat or dute).
        - **Example**:
            - **Durative Atelic**: "She **talked** for hours." → "She talked three times in one minute." (Less acceptable)
            - **Punctual Telic**: "He **clapped** his hands." → "He clapped his hands three times in one minute." (Acceptable)
    - **Note**: This method is supplementary and should be used alongside primary classification criteria.
"""

QUESTION_TEMPLATE = (
    'Please provide POS tagging for the ACTION VERBS in the following document according to my specific rules. '
    'Use the input data format to fill in the dictionary, where each key is the position of the word, and the value is a tuple containing the word and its corresponding POS tag ("dute", "pute", "duat", "puat"). For words that are not action verbs, leave the POS tag blank. '
    'Example: Input: {10: ("She", ""), 11: ("built", ""), 12: ("a", ""), 13: ("sandcastle", ""), 14: (".", "") }. '
    'Output: {10: ("She", ""), 11: ("built", "dute"), 12: ("a", ""), 13: ("sandcastle", ""), 14: (".", "")}.'
)

PROMPT_TEMPLATE = """
You are a POS (Part of Speech) tagger specialized in classifying ACTION VERBS into four categories based on their durativity and telicity. For words that are not action verbs (including modal verbs and stative verbs), leave the POS tag blank. Please return the result in JSON format only.

Context: {context}

Input:
{input_data}

Question: {question}
"""

# ----------------------- Utility Functions ----------------------- #

def clean_and_parse_json(output: str, input_data_example: Dict[int, Tuple[str, str]], logger: logging.Logger) -> Dict[int, Tuple[str, str]]:
    """
    Clean and parse the JSON output from the model.

    :param output: Raw output string from the model.
    :param input_data_example: The input data example for fallback.
    :param logger: Logger instance.
    :return: Parsed dictionary with POS tags.
    """
    try:
        # Use regex to find the first JSON object in the output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # If no JSON found, assume no verbs and set POS tags to blank
            logger.warning("No JSON found in output. Assuming no verbs present.")
            return {k: (v[0], "") for k, v in input_data_example.items()}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        logger.debug(f"Faulty output: {output}")
        # Fallback: Set POS tags to blank
        return {k: (v[0], "") for k, v in input_data_example.items()}

def reconstruct_sentence(word_dict: Dict[int, Tuple[str, str]], logger: logging.Logger) -> str:
    """
    Reconstruct the full sentence from the word dictionary.

    :param word_dict: Dictionary with word positions as keys and (word, POS) as values.
    :param logger: Logger instance.
    :return: Reconstructed sentence as a string.
    """
    try:
        # Sort the dictionary by keys (word positions)
        sorted_items = sorted(word_dict.items())
        # Concatenate the words in order
        words = [item[1][0] for item in sorted_items]
        # Join words with spaces, handling punctuation
        sentence = ''
        punctuation = {'.', ',', '!', '?', ';', ':', '"', "'", '”', '“'}
        for i, word in enumerate(words):
            if word in punctuation:
                sentence = sentence.rstrip() + word + ' '
            else:
                sentence += word + ' '
        sentence = sentence.strip()
        logger.debug(f"Reconstructed Sentence: {sentence}")
        return sentence
    except Exception as e:
        logger.error(f"Error reconstructing sentence: {e}")
        return ""


# ----------------------- Main Pipeline Function ----------------------- #

def main():
    # ----------------------- Argument Parsing ----------------------- #
    parser = argparse.ArgumentParser(description="GPT Durative/Telicity Verb Classifier Pipeline")
    
    parser.add_argument(
        '--start_book',
        type=int,
        default=DEFAULT_START_BOOK,
        help=f"Starting book number. Default: {DEFAULT_START_BOOK}"
    )
    parser.add_argument(
        '--end_book',
        type=int,
        default=DEFAULT_END_BOOK,
        help=f"Ending book number. Default: {DEFAULT_END_BOOK}"
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing separated dictionaries. Default: {DEFAULT_INPUT_DIR}"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory to save GPT-tagged Verb POS data. Default: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory to save log files. Default: {DEFAULT_LOG_DIR}"
    )
    
    args = parser.parse_args()
    
    # ----------------------- Setup Logger ----------------------- #
    logger = setup_logging(args.log_dir)
    logger.info("Starting GPT Result/ Manner Verb Classifier Pipeline")
    
    # ----------------------- Validate Arguments ----------------------- #
    if args.start_book > args.end_book:
        logger.error("START_BOOK cannot be greater than END_BOOK.")
        sys.exit(1)
    
    # ----------------------- Ensure Directories Exist ----------------------- #
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory is set to: {args.output_dir}")
    
    # ----------------------- Initialize Model and Embeddings ----------------------- #
    try:
        if MODEL.startswith("gpt"):
            logger.info(f"Initializing ChatOpenAI model: {MODEL}")
            model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL, temperature=0.0)
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            logger.info(f"Initializing Ollama model: {MODEL}")
            model = Ollama(model=MODEL)
            embeddings = OllamaEmbeddings(model=MODEL, temperature=0.0)
        logger.info("Model and embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model or embeddings: {e}")
        sys.exit(1)
    
    # ----------------------- Define Prompt Template ----------------------- #
    prompt = PromptTemplate.from_template(template=PROMPT_TEMPLATE)
    
    # Define the Processing Chain
    chain = prompt | model | StrOutputParser()
    
    # ----------------------- Iterate Over Books ----------------------- #
    for book_num in range(args.start_book, args.end_book + 1):
        logger.info(f"\nProcessing Book Number: {book_num}")
        
        # Construct Input and Output File Paths
        input_file = args.input_dir / f'document_{book_num}_sep.pkl'
        output_pickle_file = args.output_dir / f'book_part_{book_num}_dict_verb_comb.pkl'
        output_text_file = args.output_dir / f'book_part_{book_num}_dict_verb_comb.txt'  # New Text File Path
        
        # Check if Input File Exists
        if not input_file.exists():
            logger.warning(f'Input file does not exist: {input_file}. Skipping Book Number: {book_num}.')
            continue
        
        try:
            # Load the Sentences from the Input Pickle File
            with open(input_file, 'rb') as f:
                loaded_sentences = pickle.load(f)
            
            sentences = loaded_sentences
            logger.info(f'Loaded {len(sentences)} sentences from Book {book_num}.')
            
            combined_dict = {}
            for i, sentence in enumerate(sentences, 1):
                logger.info(f'Invoking chain for sentence {i} in Book {book_num}')
                input_data_example = sentence

                # Reconstruct the sentence from the dictionary
                reconstructed_sentence = reconstruct_sentence(input_data_example, logger)
                QUESTION_TEMPLATE = (
                    f'Please provide POS tagging for the VERBS in {reconstructed_sentence} according to my specific rules. '
                    'Use the input data format to fill in the dictionary, where each key is the position of the word, '
                    'and the value is a tuple containing the word and its corresponding POS tag (Manner Verb "MV" or Result Verb "RV"). '
                    'For words that are not verbs, leave the POS tag blank. Example: Input: {10: ("She", ""), 11: ("scrubbed", ""), '
                    '12: ("the", ""), 13: ("floor", ""), 14: ("until", ""), 15: ("it", ""), 16: ("shone", ""), 17: (".", "") }. '
                    'Output: {10: ("She", ""), 11: ("scrubbed", "MV"), 12: ("the", ""), 13: ("floor", ""), 14: ("until", ""), '
                    '15: ("it", ""), 16: ("shone", "RV"), 17: (".", "")}.'
                )

                input_details = {
                    "context": TAG_DESCRIPTIONS,
                    "input_data": json.dumps(input_data_example),
                    "question": QUESTION_TEMPLATE
                }
                
                try:
                    # Invoke the Processing Chain
                    output = chain.invoke(input_details)
                    logger.debug(f'Chain invoked for sentence {i} in Book {book_num}, output type: {type(output)}')
                    
                    # Parse the Output
                    parsed_sentence = clean_and_parse_json(output, input_data_example, logger)
                    combined_dict.update(parsed_sentence)
                
                except Exception as e:
                    logger.error(f'An error occurred while processing sentence {i} in Book {book_num}: {e}')
                    logger.info('Saving progress up to the last successfully processed sentence.')
                    break  # Stop processing further sentences in this book
            
            # Save the Combined Dictionary to the Output Pickle File
            with open(output_pickle_file, 'wb') as f:
                pickle.dump(combined_dict, f)
            logger.info(f'Successfully saved pickle file for Book {book_num}: {output_pickle_file}')
            
            # Save the Combined Dictionary as a JSON Text File
            with open(output_text_file, 'w', encoding='utf-8') as f:
                json.dump(combined_dict, f, ensure_ascii=False, indent=4)
            logger.info(f'Successfully saved text file for Book {book_num}: {output_text_file}\n')
        
        except Exception as e:
            logger.error(f'An unexpected error occurred while processing Book {book_num}: {e}')
            logger.info('Continuing to the next book.')
            continue
    
    logger.info('GPT Durative/Telicity Verb Classifier Pipeline processing complete.')

# ----------------------- Entry Point ----------------------- #

if __name__ == '__main__':
    main()
