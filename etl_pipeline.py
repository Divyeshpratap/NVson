# etl_pipeline.py

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import argparse
import ast
import pprint

import spacy

# ----------------------- Logging Setup ----------------------- #

def setup_logging(log_dir: Path, log_file: str = "etl_pipeline.log"):
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
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# ----------------------- Configuration Parameters ----------------------- #

DEFAULT_INPUT_FILE_PATH = Path('data/raw_texts/stories.txt')
DEFAULT_OUTPUT_POS_TAGGED_DIR = Path('data/pos_tagged/')
DEFAULT_OUTPUT_DICT_DIR = Path('data/dictionaries/')
DEFAULT_OUTPUT_SEPARATED_DIR = Path('data/separated/')
DEFAULT_REPLACEMENT_DICT = {
    # ... [Your replacement dictionary as provided earlier] ...
    "'ve": "have",
    "'m": "am",
    "Who's": "Who is",
    "’ve": "have",
    "don’t": "do not",
    "’m": "am",
    "goat’s": "goat",
    "bird's": "bird",
    "bull's": "bull",
    "bird’s": "bird",
    "hair's": "hair",
    "arm's": "arm",
    "heart’s": "heart",
    "hog's": "hog",
    "you'll": "you will",
    "it's": "it is",
    "fuller’s": "fuller",
    "lady’s": "lady",
    "didn’t": "did not",
    "It’s": "It is",
    "Craven’s": "Craven",
    "they’re": "they are",
    "I’ll": "I will",
    "don’t": "do not",
    "He’s": "He is",
    "can’t": "cannot",
    "she’ll": "she will",
    "we’ll": "we will",
    "won’t": "will not",
    "I’m": "I am",
    "isn't": "is not",
    "Heart's": "Heart",
    "jew's": "jew",
    "bee’s": "bee",
    "that’s": "that is",
    "ship’s": "ship",
    "Starbuck’s": "Starbuck",
    "whale’s": "whale",
    "n't": "not",
    "'ll": "will",
    "'d": "had or would",
    "wonder'd": "wondered",
    "'em": "them",
    "head'll": "head will",
    "'re": "are",
    "t'ain't": "it is not",
    "breakfast'll": "breakfast will",
    "o'clock": "of the clock",
    "say'st": "sayest",
    "o'er": "over",
    "tann'd": "tanned",
    "suck'd": "sucked",
    "d'ye": "do you",
    "play'd": "played",
    "ma'am": "madam",
    "'cause": "because",
    "e'en": "even",
    "p'int": "point",
    "bis'ness": "business",
    "gen'lamen": "gentlemen",
    "wot'll": "what will",
    "t'other": "the other",
    "reg'lar": "regular",
    "p'r'aps": "perhaps",
    "ta'nt": "it isn't",
    "goin'": "going",
    "ha'penny": "halfpenny",
    "gen'laman": "gentleman",
    "gen'rous": "generous",
    "nothin'": "nothing",
    "know'd": "knew",
    "doin'": "doing",
    "worn't": "was not",
    "vouldn't": "would not",
    "von't": "will not",
    "unnat'ral": "unnatural",
    "pris'ner": "prisoner",
    "soft'nin": "softening",
    "gen'lam'n": "gentleman",
    "hurra'd": "hurrahed",
    "d'israeli": "Disraeli",
    "d'alençon": "Dalencon",
    "d'anjou": "Danjou",
    "d'istria": "DIstria",
    "d'if": "DIf",
    "d'epinay": "DEpinay",
    "d'artagnan": "DArtagnan",
    "d'herblay": "DHerblay",
    "d'art": "Department of Art",
    "thriv'n": "thriving",
    "blam'd": "blamed",
    "sham'd": "shamed",
    "ne'er": "never",
    "wand'ring": "wandering",
    "crown'd": "crowned",
    "tumbl'd": "tumbled",
    "resolv'd": "resolved",
    "us'd": "used",
    "ruin'd": "ruined",
    "summon'd": "summoned",
    "nor'wester": "norwester",
    "sha'n't": "shall not",
    "o'-lantern": "of lantern",
    "pete'll": "Pete will",
    "prob'bly": "probably",
    "'bout": "about",
    "twouldn't": "it would not",
    "hain't": "have not",
    "somethin'": "something",
    "s'pose": "suppose",
    "mother'll": "mother will",
    "his'n": "his own",
    "tain't": "it is not",
    "body'd": "body would",
    "hepsey'd": "Hepsey would",
    "less'n": "less than",
    "better'n": "better than",
    "friv'lous": "frivolous",
    "sabriny'll": "Sabriny will",
    "twan't": "it was not",
    "more'n": "more than",
    "havin'": "having",
    "wa'n't": "was not",
    "calc'latin": "calculating",
    "whate'er": "whatever",
    "ye're": "you are",
    "wi'out": "without",
    "miser'ble": "miserable",
    "ol'": "old",
    "mass'chusetts": "Massachusetts",
    "nat'ral": "natural",
    "wuzn't": "was not",
    "clo'es": "clothes",
    "dagerr'otype": "daguerreotype",
    "calc'late": "calculate",
    "phot'graph": "photograph",
    "cram'bry": "cranberry",
    "gran'father": "grandfather",
    "grandma'd": "grandma would",
    "ag'in": "again",
    "meetin'-house": "meeting house",
    "s'posin": "supposing",
    "here'll": "here will",
    "y'r": "your",
    "go'n": "going",
    "twon't": "it will not",
    "on't": "on it",
    "ag'inst": "against",
    "dre'dful": "dreadful",
    "not'n": "nothing",
    "lord'll": "lord will",
    "sylvy'd": "Sylvy would",
    "cocks'-combs": "cockscombs",
    "unspurr'd": "unspurred",
    "flow'r": "flower",
    "ev'ry": "every",
    "whene'er": "whenever",
    "o'erhead": "overhead",
    "c'ck": "cock",
    "o'-pearl": "of pearl",
    "express'd": "expressed",
    "call'd": "called",
    "talk'd": "talked",
    "cover'd": "covered",
    "pr'ythee": "pray thee",
    "an'-keep": "and keep",
    "list'nin": "listening",
    "onc't": "once",
    "turn't": "turned",
    "ever'wheres": "everywhere",
    "uns'll": "ones will",
    "lightnin'-bugs": "lightning bugs",
    "he'p": "help",
    "wond'ring": "wondering",
    "call'st": "callest",
    "lov'st": "lovest",
    "need'st": "needest",
    "know'st": "knowest",
    "unta'en": "untaken",
    "cheer'ly": "cheerily",
    "fav'rite": "favorite",
    "where'er": "wherever",
    "o'erturned": "overturned",
    "e'er": "ever",
    "ca'd": "called",
    "gallows'-flesh": "gallows flesh",
    "d'aulnoy": "DAulnoy",
    "sang'st": "sangest",
    "o'toole": "OToole",
    "o'neary": "ONeary",
    "saw'st": "sawest",
    "o'ershadowed": "overshadowed",
    "o'-pearl": "opearl",
    "c'ck": "cluck"
}

# ----------------------- Utility Functions ----------------------- #

def load_text(file_path: Path, logger: logging.Logger) -> str:
    """
    Load text content from a file.

    :param file_path: Path to the input text file.
    :param logger: Logger instance.
    :return: Content of the file as a string.
    """
    try:
        logger.info(f"Loading input file: {file_path}")
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
        logger.info("Input file loaded successfully.")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decoding error for file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading file {file_path}: {e}")
        raise

def filter_lines(content: str, max_words: int, logger: logging.Logger) -> List[str]:
    """
    Filter out lines that exceed the maximum number of words.

    :param content: Original text content.
    :param max_words: Maximum allowed words per line.
    :param logger: Logger instance.
    :return: List of filtered lines.
    """
    logger.info(f"Filtering lines with more than {max_words} words.")
    lines = content.splitlines()
    filtered = [line for line in lines if len(line.split()) <= max_words]
    logger.info(f"Filtered {len(lines) - len(filtered)} lines exceeding {max_words} words.")
    return filtered

def replace_words(content: str, replacement_dict: Dict[str, str], logger: logging.Logger) -> str:
    """
    Replace words in the content based on the replacement dictionary.

    :param content: Text content.
    :param replacement_dict: Dictionary mapping original words to replacements.
    :param logger: Logger instance.
    :return: Modified text content.
    """
    logger.info("Starting word replacement.")
    # Using regex for whole word replacement to avoid partial matches
    for original, replacement in replacement_dict.items():
        pattern = re.compile(r'\b' + re.escape(original) + r'\b')
        if pattern.search(content):
            logger.debug(f"Replacing '{original}' with '{replacement}'")
            content = pattern.sub(replacement, content)
    logger.info("Word replacement completed.")
    return content

def to_lowercase(content: str, logger: logging.Logger) -> str:
    """
    Convert text content to lowercase.

    :param content: Text content.
    :param logger: Logger instance.
    :return: Lowercased text content.
    """
    logger.info("Converting content to lowercase.")
    return content.lower()

def split_into_documents(lines: List[str], max_words: int, logger: logging.Logger) -> List[str]:
    """
    Split lines into documents, each with a maximum number of words.

    :param lines: List of text lines.
    :param max_words: Maximum words per document.
    :param logger: Logger instance.
    :return: List of document strings.
    """
    logger.info(f"Splitting content into documents with up to {max_words} words each.")
    documents = []
    current_doc = []
    current_word_count = 0

    for line in lines:
        word_count = len(line.split())
        if current_word_count + word_count > max_words:
            if current_doc:
                documents.append('\n'.join(current_doc))
                logger.debug(f"Created document with {current_word_count} words.")
                current_doc = []
                current_word_count = 0
        current_doc.append(line)
        current_word_count += word_count

    if current_doc:
        documents.append('\n'.join(current_doc))
        logger.debug(f"Created final document with {current_word_count} words.")

    logger.info(f"Total documents created: {len(documents)}")
    return documents

def pos_tag_document(document: str, nlp) -> List[Dict[str, str]]:
    """
    Perform POS tagging on a single document.

    :param document: Document string.
    :param nlp: spaCy language model.
    :return: List of POS-tagged tokens as dictionaries.
    """
    spacy_doc = nlp(document)
    pos_list = [{'word': token.text, 'pos': token.tag_} for token in spacy_doc if token.text != ":" and not token.is_space]
    return pos_list

def save_pos_tagged_document_txt(pos_tags: List[Dict[str, str]], output_file: Path, logger: logging.Logger):
    """
    Save POS-tagged tokens to a text file.

    :param pos_tags: List of POS-tagged tokens.
    :param output_file: Path to the output text file.
    :param logger: Logger instance.
    """
    try:
        with output_file.open('w', encoding='utf-8') as f:
            for token in pos_tags:
                f.write(f"{token['word']}\t{token['pos']}\n")
        logger.info(f"Saved POS tags to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save POS-tagged document to {output_file}: {e}")
        raise

def save_pos_tagged_document_pickle(pos_tags: List[Dict[str, str]], output_file: Path, logger: logging.Logger):
    """
    Save POS-tagged tokens to a pickle file.

    :param pos_tags: List of POS-tagged tokens.
    :param output_file: Path to the output pickle file.
    :param logger: Logger instance.
    """
    try:
        with output_file.open('wb') as f:
            pickle.dump(pos_tags, f)
        logger.info(f"Saved POS tags to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save POS-tagged document to {output_file}: {e}")
        raise

def process_pos_tagged_files(input_dir: Path, output_dir: Path, logger: logging.Logger) -> None:
    """
    Processes .txt files in the input directory, removes POS tags, converts them into dictionaries
    with word positions as keys and (word, '') tuples as values, and saves the dictionaries
    as .txt files in the output directory.

    :param input_dir: Path to the directory containing input POS-tagged .txt files.
    :param output_dir: Path to the directory where output dictionary .txt files will be saved.
    :param logger: Logger instance.
    """
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output dictionary directory is set to: {output_dir}")

        # List all .txt files in the input directory
        input_files = list(input_dir.glob("*.txt"))
        logger.info(f"Found {len(input_files)} .txt files to process in '{input_dir}'.")

        for file_path in input_files:
            logger.info(f"Processing file: {file_path.name}")
            word_dict: Dict[int, Tuple[str, str]] = {}

            try:
                with file_path.open('r', encoding='utf-8') as infile:
                    for idx, line in enumerate(infile):
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        if '\t' not in line:
                            logger.warning(f"Skipping line {idx+1} in {file_path.name}: No tab separator found.")
                            continue
                        word, pos_tag = line.split('\t', 1)
                        word_escaped = word.replace("\\", "\\\\").replace("'", "\\'")
                        # POS tag is removed as per requirement
                        word_dict[idx] = (word_escaped, '')

                # Convert the dictionary to a formatted string
                dict_str = "{\n"
                for key in sorted(word_dict.keys()):
                    word, pos = word_dict[key]
                    dict_str += f"    {key}: ('{word}', '{pos}'),\n"
                dict_str += "}"

                # Define output file path
                output_file_name = file_path.stem + "_dict.txt"
                output_file_path = output_dir / output_file_name

                # Write the dictionary string to the output file
                with output_file_path.open('w', encoding='utf-8') as outfile:
                    outfile.write(dict_str)

                logger.info(f"Processed and saved dictionary to: {output_file_path}")

            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")

    except Exception as e:
        logger.critical(f"Critical error in processing POS-tagged files: {e}")
        raise

def is_sentence_end(word: str) -> bool:
    """
    Determines if a word signifies the end of a sentence based on punctuation.

    :param word: The word to check.
    :return: True if the word is a sentence-ending punctuation, False otherwise.
    """
    sentence_end_punctuations = {'.', '?', '!', '...', ';', ':'}
    return word in sentence_end_punctuations

def split_sentence_into_chunks(sentence_dict: Dict[int, Tuple[str, str]], max_keys: int, logger: logging.Logger) -> List[Dict[int, Tuple[str, str]]]:
    """
    Splits a single sentence dictionary into smaller chunks if it exceeds the maximum key count.

    :param sentence_dict: The original sentence dictionary to split.
    :param max_keys: Maximum number of key-value pairs per chunk.
    :param logger: Logger instance.
    :return: A list of split sentence dictionaries.
    """
    chunks = []
    current_chunk = {}
    current_count = 0

    for key in sorted(sentence_dict.keys()):
        current_chunk[key] = sentence_dict[key]
        current_count += 1

        if current_count == max_keys:
            chunks.append(current_chunk)
            logger.debug(f"Split sentence into a chunk with {current_count} keys.")
            current_chunk = {}
            current_count = 0

    if current_chunk:
        chunks.append(current_chunk)
        logger.debug(f"Created final chunk with {current_count} keys.")

    return chunks

def create_split_dictionaries(original_dict: Dict[int, Tuple[str, str]], min_keys: int, max_keys: int, logger: logging.Logger) -> List[Dict[int, Tuple[str, str]]]:
    """
    Splits the original dictionary into multiple dictionaries based on the rules.

    :param original_dict: The original dictionary to split.
    :param min_keys: Minimum number of key-value pairs per split dictionary.
    :param max_keys: Maximum number of key-value pairs per split dictionary.
    :param logger: Logger instance.
    :return: A list of split dictionaries.
    """
    sentences = []
    current_sentence = {}

    # Identify sentences based on sentence-ending punctuation
    for key in sorted(original_dict.keys()):
        current_sentence[key] = original_dict[key]
        word, _ = original_dict[key]
        if is_sentence_end(word):
            sentences.append(current_sentence)
            current_sentence = {}

    # Add any remaining words as the last sentence
    if current_sentence:
        sentences.append(current_sentence)

    split_dicts = []
    current_group = {}
    current_count = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If sentence itself exceeds max_keys, split it
        if sentence_length > max_keys:
            if current_group:
                split_dicts.append(current_group)
                logger.debug(f"Finalizing current group with {current_count} keys before splitting a long sentence.")
                current_group = {}
                current_count = 0

            # Split the long sentence into chunks
            sentence_chunks = split_sentence_into_chunks(sentence, max_keys, logger)
            split_dicts.extend(sentence_chunks)
            logger.debug(f"Split a long sentence into {len(sentence_chunks)} chunks.")
            continue

        # Check if adding this sentence would exceed max_keys
        if current_count + sentence_length > max_keys:
            if current_group:
                split_dicts.append(current_group)
                logger.debug(f"Finalized a group with {current_count} keys before adding a new sentence.")
            # Start a new group
            current_group = sentence.copy()
            current_count = sentence_length
        else:
            # Add sentence to the current group
            current_group.update(sentence)
            current_count += sentence_length

    # Finalize the last group
    if current_group:
        split_dicts.append(current_group)
        logger.debug(f"Finalized the last group with {current_count} keys.")

    # Ensure each split has at least min_keys
    final_split_dicts = []
    temp_group = {}
    temp_count = 0

    for split in split_dicts:
        split_length = len(split)

        if split_length >= min_keys:
            if temp_group:
                # Try to combine temp_group with the current split
                if temp_count + split_length <= max_keys:
                    temp_group.update(split)
                    temp_count += split_length
                    logger.debug(f"Combined split of {split_length} keys with temp_group of {temp_count - split_length} keys.")
                else:
                    final_split_dicts.append(temp_group)
                    logger.debug(f"Added temp_group with {temp_count} keys to final_split_dicts before starting a new group.")
                    temp_group = split.copy()
                    temp_count = split_length
            else:
                final_split_dicts.append(split)
                logger.debug(f"Added split of {split_length} keys directly to final_split_dicts.")
        else:
            # Split is less than min_keys, combine it with temp_group
            if temp_count + split_length <= max_keys:
                temp_group.update(split)
                temp_count += split_length
                logger.debug(f"Combined small split of {split_length} keys with temp_group.")
            else:
                if temp_group:
                    final_split_dicts.append(temp_group)
                    logger.debug(f"Added temp_group with {temp_count} keys to final_split_dicts before adding small split.")
                temp_group = split.copy()
                temp_count = split_length

    if temp_group:
        final_split_dicts.append(temp_group)
        logger.debug(f"Added the remaining temp_group with {temp_count} keys to final_split_dicts.")

    return final_split_dicts

def process_split_dictionaries(input_dir: Path, output_dir: Path, min_keys: int, max_keys: int, logger: logging.Logger) -> None:
    """
    Processes split dictionaries from the input directory and saves them to the output directory.

    :param input_dir: Path to the directory containing input split dictionaries (.txt files).
    :param output_dir: Path to the directory where output split dictionaries will be saved.
    :param min_keys: Minimum number of key-value pairs per split dictionary.
    :param max_keys: Maximum number of key-value pairs per split dictionary.
    :param logger: Logger instance.
    """
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory for separated dictionaries is set to: {output_dir}")

        # List all _dict.txt files in the input directory
        input_files = list(input_dir.glob("*_dict.txt"))
        logger.info(f"Found {len(input_files)} _dict.txt files to process in '{input_dir}'.")

        for file_path in input_files:
            logger.info(f"Processing split for file: {file_path.name}")
            try:
                with file_path.open('r', encoding='utf-8') as infile:
                    dict_content = infile.read()
                    original_dict = ast.literal_eval(dict_content)
                logger.debug(f"Loaded dictionary from {file_path.name} with {len(original_dict)} keys.")
            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")
                continue

            # Split the dictionary based on the rules
            split_dicts = create_split_dictionaries(original_dict, min_keys, max_keys, logger)
            logger.info(f"File {file_path.name} split into {len(split_dicts)} parts.")

            # Define base name for output files
            base_name = file_path.stem.replace("_dict", "")

            # Save split dictionaries as a pickle file
            output_pkl_path = output_dir / f"{base_name}_sep.pkl"
            try:
                with output_pkl_path.open('wb') as outfile_pkl:
                    pickle.dump(split_dicts, outfile_pkl)
                logger.info(f"Saved split dictionaries to pickle file: {output_pkl_path.name}")
            except Exception as e:
                logger.error(f"Error writing pickle file {output_pkl_path.name}: {e}")

            # Save split dictionaries as a formatted text file
            output_txt_path = output_dir / f"{base_name}_sep.txt"
            try:
                with output_txt_path.open('w', encoding='utf-8') as outfile_txt:
                    # Use pprint to format the list of dictionaries
                    pprint.pprint(split_dicts, stream=outfile_txt, width=120, compact=False)
                logger.info(f"Saved split dictionaries to text file: {output_txt_path.name}")
            except Exception as e:
                logger.error(f"Error writing text file {output_txt_path.name}: {e}")

    except Exception as e:
        logger.critical(f"Critical error in processing split dictionaries: {e}")
        raise

# ----------------------- Main Pipeline Function ----------------------- #

def main():
    # ----------------------- Argument Parsing ----------------------- #
    parser = argparse.ArgumentParser(description="ETL Pipeline for Custom POS Tagger")
    
    # Parameters from ETL.py
    parser.add_argument(
        '--input_file_path',
        type=Path,
        default=DEFAULT_INPUT_FILE_PATH,
        help=f"Path to the input text file. Default: {DEFAULT_INPUT_FILE_PATH}"
    )
    parser.add_argument(
        '--max_line_words',
        type=int,
        default=100,
        help="Maximum number of words per line. Default: 100"
    )
    parser.add_argument(
        '--max_document_words',
        type=int,
        default=2000,
        help="Maximum number of words per document. Default: 2000"
    )
    
    # Parameters from ETL2.py
    parser.add_argument(
        '--output_pos_tagged_dir',
        type=Path,
        default=DEFAULT_OUTPUT_POS_TAGGED_DIR,
        help=f"Directory to save POS-tagged documents. Default: {DEFAULT_OUTPUT_POS_TAGGED_DIR}"
    )
    parser.add_argument(
        '--output_dict_dir',
        type=Path,
        default=DEFAULT_OUTPUT_DICT_DIR,
        help=f"Directory to save dictionary files. Default: {DEFAULT_OUTPUT_DICT_DIR}"
    )
    parser.add_argument(
        '--output_separated_dir',
        type=Path,
        default=DEFAULT_OUTPUT_SEPARATED_DIR,
        help=f"Directory to save separated dictionaries. Default: {DEFAULT_OUTPUT_SEPARATED_DIR}"
    )
    parser.add_argument(
        '--min_keys_per_split',
        type=int,
        default=20,
        help="Minimum number of key-value pairs per split dictionary. Default: 20"
    )
    parser.add_argument(
        '--max_keys_per_split',
        type=int,
        default=60,
        help="Maximum number of key-value pairs per split dictionary. Default: 60"
    )
    
    # Additional Parameters
    parser.add_argument(
        '--log_dir',
        type=Path,
        default=Path('logs/'),
        help="Directory to save log files. Default: logs/"
    )
    
    args = parser.parse_args()
    
    # ----------------------- Setup Logger ----------------------- #
    logger = setup_logging(args.log_dir)
    logger.info("Starting ETL Pipeline")
    
    # ----------------------- Load spaCy Model ----------------------- #
    try:
        logger.info("Loading spaCy model 'en_core_web_trf'.")
        nlp = spacy.load("en_core_web_trf")
        logger.info("spaCy model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        raise
    
    # ----------------------- Step 1: Extract and Initial Preprocessing ----------------------- #
    try:
        # Load the input file
        content = load_text(args.input_file_path, logger)
        
        # Filter lines exceeding max word count
        filtered_lines = filter_lines(content, args.max_line_words, logger)
        
        # Replace unusual words
        replaced_content = replace_words('\n'.join(filtered_lines), DEFAULT_REPLACEMENT_DICT, logger)
        
        # Convert to lowercase
        lowercased_content = to_lowercase(replaced_content, logger)
        
        # Split into documents
        split_lines = lowercased_content.splitlines()
        documents = split_into_documents(split_lines, args.max_document_words, logger)
        
    except Exception as e:
        logger.critical(f"Error during extraction and preprocessing: {e}")
        raise
    
    # ----------------------- Step 2: POS Tagging and Saving ----------------------- #
    try:
        logger.info("Starting POS tagging and saving process.")
        args.output_pos_tagged_dir.mkdir(parents=True, exist_ok=True)
        
        total_docs = len(documents)
        logger.info(f"Total documents to process: {total_docs}")
        
        for idx, document in enumerate(documents, 1):
            logger.info(f"Processing document {idx}/{total_docs}")
            pos_tags = pos_tag_document(document, nlp)
            
            # Define the output file names
            output_txt_file = args.output_pos_tagged_dir / f'document_{idx}.txt'
            output_pickle_file = args.output_pos_tagged_dir / f'document_{idx}.pickle'
            
            # Save the POS-tagged document as .txt
            save_pos_tagged_document_txt(pos_tags, output_txt_file, logger)
            
            # Save the POS-tagged document as .pickle
            save_pos_tagged_document_pickle(pos_tags, output_pickle_file, logger)
        
        logger.info("POS tagging and saving completed successfully.")
        
    except Exception as e:
        logger.critical(f"Error during POS tagging and saving: {e}")
        raise
    
    # ----------------------- Step 3: Dictionary Creation and Splitting ----------------------- #
    try:
        logger.info("Starting dictionary creation and splitting process.")
        
        # Process POS-tagged files to create dictionaries without POS tags
        process_pos_tagged_files(args.output_pos_tagged_dir, args.output_dict_dir, logger)
        
        # Split the dictionaries into smaller chunks
        process_split_dictionaries(
            input_dir=args.output_dict_dir,
            output_dir=args.output_separated_dir,
            min_keys=args.min_keys_per_split,
            max_keys=args.max_keys_per_split,
            logger=logger
        )
        
        logger.info("Dictionary creation and splitting completed successfully.")
        
    except Exception as e:
        logger.critical(f"Error during dictionary creation and splitting: {e}")
        raise
    
    logger.info("ETL Pipeline completed successfully.")

# ----------------------- Entry Point ----------------------- #

if __name__ == '__main__':
    main()
