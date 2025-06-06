PROMPT_OPEN = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    Question:\n {question} \n\n

    Answer:
"""

PROMPT_OPEN_RAG = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    Answer the question using the provided context.\n\n
    Context: {context} \n\n

    Question:\n {question} \n\n

    Answer:
"""

PROMPT_OPEN_BASELINE = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    The answer must be based on the following reasoning path: {path} and the context associated: {text}\n\n

    Question:\n {question} \n\n

    Answer:
"""

PROMPT_OPEN_BASELINE_TEXT = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    The answer must be based on the following context associated: {text}\n\n

    Question:\n {question} \n\n

    Answer:
"""

PROMPT_OPEN_BASELINE_PATH = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    The answer must be based on the following reasoning path: {path}.\n\n

    Question:\n {question} \n\n

    Answer:
"""

PROMPT_QUIZ_BASELINE = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    The choice of the correct answer should be based on the following sequence of reasoning: {path} and the context associated: {text}\n\n
    Question:\n {question}\n\n
    Options:\n
    A. {o1}\n
    B. {o2}\n
    C. {o3}\n
    D. {o4}\n
    E. {o5}\n\n
    The answer MUST be a single letter.\n
    Correct option:
"""

PROMPT_QUIZ_BASELINE_PATH = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    The choice of the correct answer should be based on the following sequence of reasoning: {path}.\n\n
    Question:\n {question}\n\n
    Options:\n
    A. {o1}\n
    B. {o2}\n
    C. {o3}\n
    D. {o4}\n
    E. {o5}\n\n
    The answer MUST be a single letter.\n
    Correct option:
"""

PROMPT_QUIZ_BASELINE_TEXT = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    The choice of the correct answer should be based on the following context associated: {text}\n\n
    Question:\n {question}\n\n
    Options:\n
    A. {o1}\n
    B. {o2}\n
    C. {o3}\n
    D. {o4}\n
    E. {o5}\n\n
    The answer MUST be a single letter.\n
    Correct option:
"""

PROMPT_QUIZ = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    Question:\n {question}\n\n
    Options:\n
    A. {o1}\n
    B. {o2}\n
    C. {o3}\n
    D. {o4}\n
    E. {o5}\n\n
    The answer MUST be a single letter.\n
    Correct option:
"""

PROMPT_QUIZ_RAG = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    Answer the question using the provided context.\n\n
    Context: {context}\n\n
    Question:\n {question}\n\n
    Options:\n
    A. {o1}\n
    B. {o2}\n
    C. {o3}\n
    D. {o4}\n
    E. {o5}\n\n
    The answer MUST be a single letter.\n
    Correct option:
"""