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

# PROMPT_OPEN_ONE = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
#     - be precise;\n
#     - no longer than 100/150 characters;\n
#     - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

#     This is an example of question and answer: \n
#     "Question: A patient with noncardiac chest pain and normal coronary arteries fails to respond to a trial of high-dose proton pump inhibitors. Endoscopy reveals an abnormality. How does this finding alter the diagnostic approach and subsequent treatment strategy for their chest pain?
#     Answer: The discovery of an abnormality during endoscopy indicates that the chest pain is likely due to an esophageal motility disorder rather than GERD. Further investigation and treatment will be tailored to the specific abnormality found, such as achalasia or diffuse esophageal spasm, 
#         rather than focusing solely on acid reflux."\n\n

#     Question: {question} \n\n

#     Answer:
# """

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

CHECK_QUESTION = """I will give you a quiz about how to manage a patient affected by {condition}, with some option and the real answer. 
    your task is to check if the quiz makes sense and, if there is more than one correct option.\n\n
    Question: {question}\n\n
    Options: {options}\n
    Correct option: {correct_option}\n\n
    Your answer must follow the following rules:\n
    1) answer with the word "CORRECT" if the quiz makes sense and there is only one correct option;\n
    2) answer with the word "WRONG" if the quiz does not make sense or if there is more than one correct option.\n\n
    The answer MUST be one word among those described above, without any additional text!\n
    Answer: 
"""

CHECK_QUESTION_RATE = """I will give you a quiz about how to manage a patient affected by {condition}, with some option and the real answer. 
    your task is to check if the quiz makes sense and if there is more than one correct option.\n\n
    Evaluate the questions and the answer using also the provided context.\n
    Context: {context}\n\n
    Question: {question}\n\n
    Options: {options}\n
    Correct option: {correct_option}\n\n
    Give a score from 1 to 5 on how sensible, clear and correctly answered the question is with the given options. 
    If there are several plausible options correct for the question, penalize the score.\n\n
    The answer MUST be a number between 1 and 5, without any additional text!\n
    Answer: 
"""

CHECK_QUESTION_OPIONS = """I have this question: {question}\n
    The question is based on the following sequence of reasoning: {path} for this condition: {cond}\n
    This are the options to choose: {options}\n
    Which is the correct option? The answer MUST be one of the options described above, without any additional text!\n
"""

PROMPT_QUIZ_MEDQA = """The following is a multiple-choice question about how to manage a patient. 
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

PROMPT_QUIZ_RAG_MEDQA = """The following is a multiple-choice question about how to manage a patient. 
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