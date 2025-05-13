import re
from collections import Counter
import glob
import os
from typing import Dict, List

def extract_option(answer: str):
    """Extract the chosen option (A, B, C, D, E or a, b, c, d, e) from the LLM's answer."""
    
    answer = str(answer).strip()  # Remove leading/trailing spaces
    answer = answer.replace("**", "")

    match = re.fullmatch(r'[a-eA-E]', answer)
    if match:
        return match.group(0).upper()

    match = re.search(r'^([a-eA-E])\s', answer)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'^([a-eA-E])\.', answer)
    if match:
        return match.group(1).upper()

    match = re.search(r'\(([a-eA-E])\)|\s([a-eA-E])\)', answer)
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    match = re.match(r'^[A-E]([A-Z])', answer)
    if match:
        return match.group(0)[0]

    match = re.match(r'^[a-eA-E][^a-zA-Z]', answer)
    if match:
        return match.group(0)[0]

    match = re.search(r'the correct answer is[:\s]*\(?([a-eA-E])\)?', answer.lower(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'the answer is[:\s]*\(?([a-eA-E])\)?', answer.lower(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'Option *\(?([a-eA-E])\)?', answer.lower(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'Answer: *\(?([a-eA-E])\)?', answer.lower(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'[:\.,]\s*([a-eA-E])$', answer)
    if match:
        return match.group(1).upper()

    match = re.match(r'^([a-eA-E])[A-Z]', answer)
    if match:
        return match.group(1).upper()
    
    return None

def check_options(df):
    """Check if the options are consistent across all questions."""
    options = []
    none_values = []
    
    for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
        for original_value, extracted in zip(df[col], df[col].apply(extract_option).tolist()):
            if extracted is None:
                none_values.append(original_value)
            else:
                options.append(extracted)
    
    return Counter(options), none_values

def check_results(search_directory: str, string_to_check: str, search_strings: List[str]) -> List[str]:
    # Get all matching files
    matching_files = glob.glob(os.path.join(search_directory, string_to_check))

    remaining_strings = [name for name in search_strings if not any(name.replace(":", "_") in os.path.basename(file) for file in matching_files)]

    print("Remaining models to run: ", remaining_strings)

    return remaining_strings