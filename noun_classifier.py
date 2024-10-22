# gpt_noun_classifier.py

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

def setup_logging(log_dir: Path, log_file: str = "gpt_noun_classifier.log"):
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
DEFAULT_OUTPUT_DIR = Path('data/gpt_tagged/noun/')
DEFAULT_LOG_DIR = Path('logs/')

DEFAULT_START_BOOK = 1
DEFAULT_END_BOOK = 2

MODEL = 'gpt-4o-mini'  # Options: 'gpt-4o-mini', 'llama3.1:70b', 'llama2:latest'

# Tag Descriptions and Prompt Template
TAG_DESCRIPTIONS = """
I need you to classify nouns into two categories: Shape Nouns (SN) and Non-Shape Nouns (NSN). Please disregard any prior knowledge you may have about these nouns, as I want to create a novel POS tagging system that classifies nouns by a toddler's perception of how they remember new nouns (shape or not shape), rather than traditional categories like proper, common, singular, plural, etc. To provide some context, toddlers often learn new words based on shape similarity. For instance, if a toddler (12-24 months old) sees a banana, they can recognize another banana due to its similar shape. However, if they see butter, they might struggle to form a similar connection since butter can change shape, and toddlers tend to understand it more as a material or taste-based noun. With this in mind, I want you to approach the classification process as a toddler would when learning new words.

Functionality Exclusion: Nouns should be classified based on countability, solidity, material/mass, and toddler perception only. For example, you cannot classify "train," "tractor," "watch," etc., as Non-Shape Nouns simply because they are recognized by their functionality. This is because children do not perceive objects through their functional use but rather through their physical attributes. Therefore, if an object has both a functional and a distinct shape, it should still be classified as a Shape Noun, even if its primary purpose is functional.

Examples:
- "Apple" is a Shape Noun (SN) because of its consistent round shape, countability, and solid form which is not easily deformable.
- "Water" is a Non-Shape Noun (NSN) because it takes the shape of its container and uses mass or material syntax.
- "Horse" is a Shape Noun (SN) because it is a definite animal, and when a toddler sees another horse, they will recognize it based on the horse's standard shape.

Sometimes context can also influence if a noun is shape or non shape. For example:
Context-Dependent Nouns: These are nouns whose classification can be affected by context. For instance, "spring" can be a Shape Noun in "The spring in the watch" but a Non-Shape Noun in "I love taking walks in the spring."
"""

QUESTION_TEMPLATE = (
    'Please provide POS tagging for the NOUNS in the following document according to my specific rules. '
    'Use the input data format to fill in the dictionary, where each key is the position of the word, and the value is a tuple containing the word and its corresponding POS tag (Shape Noun "SN" or Non-Shape Noun "NSN"). '
    'For words that are not nouns, leave the POS tag blank. '
    'Example: Input: {10: ("She", ""), 11: ("plays", ""), 12: ("piano", ""), 13: ("with", ""), 14: ("fingers", ""), 15: ("and", ""), 16: ("eats", ""), 17: ("cheese", ""), 18: ("with", ""), 19: ("Joey", ""), 20: (".", "") }. '
    'Output: {10: ("She", ""), 11: ("plays", ""), 12: ("piano", "SN"), 13: ("with", ""), 14: ("fingers", "SN"), 15: ("and", ""), 16: ("eats", ""), 17: ("cheese", "NSN"), 18: ("with", ""), 19: ("Joey", "NSN"), 20: (".", "")}.'
)

PROMPT_TEMPLATE = """
You are a POS (Part of Speech) tagger. Your task is to assign the appropriate POS tags to NOUN words in the input dictionary. For words that are not nouns, please leave the POS tag blank. Only tag nouns according to the specific context provided below. Ensure you use the exact tag descriptions ("SN" for Shape Nouns, "NSN" for Non-Shape Nouns) and apply the tagging rules correctly.

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
            # If no JSON found, assume no nouns and set POS tags to blank
            logger.warning("No JSON found in output. Assuming no nouns present.")
            return {k: (v[0], "") for k, v in input_data_example.items()}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        logger.debug(f"Faulty output: {output}")
        # Fallback: Set POS tags to blank
        return {k: (v[0], "") for k, v in input_data_example.items()}

# ----------------------- Main Pipeline Function ----------------------- #

def main():
    # ----------------------- Argument Parsing ----------------------- #
    parser = argparse.ArgumentParser(description="GPT Noun Classifier Pipeline")
    
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
        help=f"Output directory to save GPT-tagged POS data. Default: {DEFAULT_OUTPUT_DIR}"
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
    logger.info("Starting GPT Noun Classifier Pipeline")
    
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
        output_pickle_file = args.output_dir / f'book_part_{book_num}_dict_noun_comb.pkl'
        output_text_file = args.output_dir / f'book_part_{book_num}_dict_noun_comb.txt'  # New Text File Path
        
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
    
    logger.info('GPT Noun Classifier Pipeline processing complete.')

# ----------------------- Entry Point ----------------------- #

if __name__ == '__main__':
    main()
