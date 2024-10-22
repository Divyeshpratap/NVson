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
DEFAULT_OUTPUT_DIR = Path('data/gpt_tagged/verb/')
DEFAULT_LOG_DIR = Path('logs/')

DEFAULT_START_BOOK = 1
DEFAULT_END_BOOK = 2

MODEL = 'gpt-4o-mini'  # Options: 'gpt-4o-mini', 'llama3.1:70b', 'llama2:latest'

# Tag Descriptions and Prompt Template
TAG_DESCRIPTIONS = """
**Verb Classification: Manner Verb (MV) vs. Result Verb (RV)**
You are tasked with classifying verbs in a sentence into two distinct categories based on their semantic roles:
- **Manner Verbs (MV)**:
    - **Definition**: These verbs encode the *how* of an action, focusing on the method or pattern without specifying an outcome.
    - **Semantic Basis**: Involve **nonscalar** or **complex** changes like multidimensional actions (e.g., the specific pattern of leg movements while jogging)
    - **Examples**: nibble, rub, scribble, sweep, flutter, laugh, run, swim.
    - **Usage**: "She **scribbled** on the notebook." (Focus on the writing method)

- **Result Verbs (RV)**:
    - **Definition**: These verbs encode the *outcome* or resultant state that follows from an action.
    - **Semantic Basis**: Involve **scalar** changes like changes along a defined scale (e.g., temperature increasing, distance decreasing). 
    - **Examples**: clean, cover, empty, fill, freeze, kill, melt, open, arrive, die, enter, faint.
    - **Usage**: "He **melted** the ice." (Focus on the ice turning to water)

**Key Guidelines**:

1. **Exclusive Classification**: Each verb is either MV or RV based on its primary lexical meaning.
2. **Polysemy Handling**: Some verbs have multiple senses; classify them based on context.
    - **Example**:
        - **MV**: "She **cuts** carefully with the knife."
        - **RV**: "Dana **cut** the rope."
3. **Context-Dependent Classification**:
    - **"Wipe"**:
        - **MV**: "He **wiped** the table carefully."
        - **RV**: "He **wiped** the table clean."
    - **"Paint"**:
        - **MV**: "She **painted** the wall slowly."
        - **RV**: "She **painted** the wall red."
4. **Comprehensive Tagging**:
    - **Instruction**: Don't forget to also tag **all** action verbs in the sentence as either MV or RV based on their semantic roles.
    - **Example**:
        - "She **asked** a question." → "asked" should be tagged as **MV**.
        - "He **said** quietly." → "said" should be tagged as **MV**.
5. **Supplementary Classification Method**:
    - **"But Nothing Changed"** Sentence Frame:
        - **Purpose**: Provides an additional method to determine verb classification.
        - **Guidelines**:
            - If adding "but nothing changed" makes the sentence unacceptable, classify the verb as a Result Verb (RV).
            - If adding "but nothing changed" keeps the sentence acceptable, classify the verb as a Manner Verb (MV).
    - **Example**:
        - **Sentence 1 (RV):** "She **became** a teacher."
            - **Modified Sentence:** "She became a teacher, but nothing changed."
            - **Result:** seems unacceptable, making it a result verb: something changed because now she is a teacher.
        - **Sentence 2 (MV):** "She **walked** to the store."
            - **Modified Sentence:** "She walked to the store, but nothing changed."
            - **Result:** here the sentence seems acceptable, making it a manner verb
        - **Note**: This method is supplementary and should be used alongside primary classification criteria.
"""

QUESTION_TEMPLATE = (
    'Please provide POS tagging for the VERBS in the following document according to my specific rules. Use the input data format to fill in the dictionary, where each key is the position of the word, and the value is a tuple containing the word and its corresponding POS tag (Manner Verb "MV" or Result Verb "RV"). For words that are not verbs, leave the POS tag blank. Example: Input: {10: ("She", ""), 11: ("scrubbed", ""), 12: ("the", ""), 13: ("floor", ""), 14: ("until", ""), 15: ("it", ""), 16: ("shone", ""), 17: (".", "") }. Output: {10: ("She", ""), 11: ("scrubbed", "MV"), 12: ("the", ""), 13: ("floor", ""), 14: ("until", ""), 15: ("it", ""), 16: ("shone", "RV"), 17: (".", "")}.'
)

PROMPT_TEMPLATE = """
You are a POS (Part of Speech) tagger specialized in classifying verbs as Manner Verbs (MV) or Result Verbs (RV) based on specific semantic criteria. For words that are not verbs, leave the POS tag blank. But please make sure to classify all verbs (except Modal verbs) including action verb, though something like "is/ has, would, etc." has to be skipped.

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

# ----------------------- Main Pipeline Function ----------------------- #

def main():
    # ----------------------- Argument Parsing ----------------------- #
    parser = argparse.ArgumentParser(description="GPT Result/ Manner Verb Classifier Pipeline")
    
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
    
    logger.info('GPT Result/ Manner Verb Classifier Pipeline processing complete.')

# ----------------------- Entry Point ----------------------- #

if __name__ == '__main__':
    main()
