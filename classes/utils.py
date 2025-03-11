import re
from collections import Counter

def extract_option(answer):
    """Extract the chosen option (A, B, C, D, E or a, b, c, d, e) from the LLM's answer."""
    
    answer = str(answer).strip()  # Remove leading/trailing spaces
    answer = answer.replace("**", "")

    # 3) Match a single character (a-e or A-E)
    match = re.fullmatch(r'[a-eA-E]', answer)
    if match:
        return match.group(0).upper()

    # 4) Match a letter followed by a space (with or without more text)
    match = re.search(r'^([a-eA-E])\s', answer)
    if match:
        return match.group(1).upper()
    
    # 5) Match a letter followed by a period, e.g., A., b.
    match = re.search(r'^([a-eA-E])\.', answer)
    if match:
        return match.group(1).upper()

    # 6) Match a letter inside parentheses, e.g., (a), (B),
    # OR a letter preceded by a space and followed by `)`, e.g., "text d) more text" (but NOT "textd)")
    match = re.search(r'\(([a-eA-E])\)|\s([a-eA-E])\)', answer)
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    # 1) Check for uppercase letter A-E followed by another uppercase letter (A-E)
    match = re.match(r'^[A-E]([A-Z])', answer)
    if match:
        return match.group(0)[0]

    # 2) Check for lowercase letter a-e or uppercase letter A-E followed by a non-letter character
    match = re.match(r'^[a-eA-E][^a-zA-Z]', answer)
    if match:
        return match.group(0)[0]

    # Handle cases like "the correct answer is:a", "the correct answer is: a", etc.
    match = re.search(r'the correct answer is[:\s]*\(?([a-eA-E])\)?', answer.lower(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # NEW RULE: Match a letter a-e or A-E followed by any uppercase letter
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