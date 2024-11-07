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
DEFAULT_OUTPUT_DIR = Path('data/gpt_tagged/countSolidMaterial/')
DEFAULT_LOG_DIR = Path('logs/')

DEFAULT_START_BOOK = 11
DEFAULT_END_BOOK = 15

MODEL = 'gpt-4o-mini'  # Options: 'gpt-4o-mini', 'llama3.1:70b', 'llama2:latest'

# Tag Descriptions and Prompt Template
TAG_DESCRIPTIONS = """
**Definitions:**
- **Concrete Noun:** Nouns that refer to tangible, physical objects that can be perceived through the senses.
- **Abstract Noun:** Nouns that refer to ideas, qualities, or states rather than physical or tangible objects.

- **Solid Noun:** Concrete Nouns that name solid, cohesive, and bounded objects, typically organized by shape and can be pluralized.
- **Count Noun:** Concrete Nouns that can take plural forms and refer to discrete, countable **objects**.
- **Material Noun:** Concrete Nouns that are organized based on material properties. They usually refer to non-countable, continuous substances and cannot be pluralized.


I need you to classify nouns based on three categories: Countable, Material, and Solidity. The classification should result in one of the following tags:

1. **socoma**: Solid, Countable, Material
   - *Example:* "pretzel", "glass"
   
2. **soco**: Solid, Countable, Not Material
   - *Example:* "bird", "muffin"
   
3. **soma**: Solid, Not Countable, Material
   - *Example:* "French Fries"
   
4. **so**: Solid, Not Countable, Not Material
   - *Example:* "corn"
   
5. **coma**: Not Solid, Countable, Material
   - *Example:* "Snowman"
   
6. **co**: Not Solid, Countable, Not Material
   - *Example:* "Pillow"
   
7. **ma**: Not Solid, Not Countable, Material
   - *Example:* "jelly"
   
8. **neither**: Not Solid, Not Countable, Not Material, and any Intangible Noun.
   - *Example:* "circus", "bubbles", "night"

**Note:** 
- **Abstract or Intangible Nouns** (e.g., "weeks," "freedom", etc.) should be classified as **neither**, even if they can be pluralized.   

**Context-Dependent Nouns:**
Some nouns may change their classification based on context. For example, "water" can be:
- **Countable** in "Give me two waters." (referring to servings)
- **Not Countable** in "Give me a glass of water." (referring to the substance)

**Input Format:**
- The input is a dictionary where each key is the position of the word in the document, and the value is a tuple containing the word and an empty tag which you need to fill if the word is a Noun.
- Example Input:
```json
{
    10: ("She", ""),
    11: ("plays", ""),
    12: ("piano", ""),
    13: ("with", ""),
    14: ("fingers", ""),
    15: ("and", ""),
    16: ("eats", ""),
    17: ("cheese", ""),
    18: ("with", ""),
    19: ("Joey", ""),
    20: (".", "")
}
"""

PROMPT_TEMPLATE = """
You are a POS (Part of Speech) tagger. Your task is to assign the appropriate tags to NOUN words in the input dictionary based on the specific tagging rules provided below. For words that are not nouns, please leave the tag blank. Please return the result in JSON only

**Tagging Rules:**
{context}

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

def reconstruct_sentence(word_dict) -> str:
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
        
        return sentence
    except Exception as e:
        print("Error")
        return ""

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
        output_pickle_file = args.output_dir / f'book_part_{book_num}_dict_nounProperties_comb.pkl'
        output_text_file = args.output_dir / f'book_part_{book_num}_dict_nounProperties_comb.txt'  # New Text File Path
        
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
                reconstructed_sentence = reconstruct_sentence(input_data_example)
                myQuestion = (
                    f'Please provide tagging for only TANGIBLE NOUNS in the input sentence: "{reconstructed_sentence}" '
                    'according to my specific rules. Use the input data format to fill in the dictionary, '
                    'where each key is the position of the word, and the value is a tuple containing the word '
                    'and its corresponding tag ("socoma", "soco", "soma", "so", "coma", "co", "ma", "neither"). '
                    'For words that are not nouns, leave the tag blank. Example: Input: {10: ("She", ""), 11: ("plays", ""), '
                    '12: ("piano", ""), 13: ("with", ""), 14: ("fingers", "")}. Output: '
                    '{10: ("She", ""), 11: ("plays", ""), 12: ("piano", "soco"), 13: ("with", ""), 14: ("fingers", "co")}.'
                )
                input_details = {
                    "context": TAG_DESCRIPTIONS,
                    "input_data": json.dumps(input_data_example),
                    "question": myQuestion
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
