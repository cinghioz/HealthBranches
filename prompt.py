PROMPT_OPEN = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - be precise;\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    Question: {question} \n\n

    Answer:
"""

PROMPT_OPEN_ONE = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - be precise;\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    This is an example of question and answer: \n
    "Question: A patient with noncardiac chest pain and normal coronary arteries fails to respond to a trial of high-dose proton pump inhibitors. Endoscopy reveals an abnormality. How does this finding alter the diagnostic approach and subsequent treatment strategy for their chest pain?
    Answer: The discovery of an abnormality during endoscopy indicates that the chest pain is likely due to an esophageal motility disorder rather than GERD. Further investigation and treatment will be tailored to the specific abnormality found, such as achalasia or diffuse esophageal spasm, 
        rather than focusing solely on acid reflux."\n\n

    Question: {question} \n\n

    Answer:
"""

PROMPT_OPEN_RAG = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - be precise;\n
    - no longer than 100/150 characters;\n
    - must contain a decision made on the decision-making process of a clinical case described in the question.\n\n

    Answer the question using the provided context.\n\n
    Context: {context} \n\n

    Question: {question} \n\n

    Answer:
"""

PROMPT_OPEN_BASELINE = """Answer the question about how to manage a patient affected by {condition}. The answer must:\n
    - be precise;\n
    - no longer than 100/150 characters;\n
    - must contain some sort of decision-making process in to it;\n
    - must highlight which are the steps a physician would take;\n
    - what the results will indicate.\n\n

    The answer must be based on the following reasoning path: {path} and the context associated: {text}\n\n

    Question: {question} \n\n

    Answer:
"""

PROMPT_QUIZ_BASELINE = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    The choice of the correct answer should be based on the following sequence of reasoning: {path} and the context associated: {text}\n\n
    Question: {question}\n\n
    Choices:\n
    a. {o1}\n
    b. {o2}\n
    c. {o3}\n
    d. {o4}\n
    e. {o5}\n\n
    Answer:
"""

PROMPT_QUIZ = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    Question: {question}\n\n
    Choices:\n
    a. {o1}\n
    b. {o2}\n
    c. {o3}\n
    d. {o4}\n
    e. {o5}\n\n
    Answer:
"""

PROMPT_QUIZ_RAG = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply ONLY with the letter (A,B,C,D,E) of the answer you think is CORRECT, without any additional text!\n\n
    Answer the question using the provided context.\n\n
    Context: {context}\n\n
    Question: {question}\n\n
    Choices:\n
    a. {o1}\n
    b. {o2}\n
    c. {o3}\n
    d. {o4}\n
    e. {o5}\n\n
    Answer:
"""

PROMPT_MED = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply only with the letter of the correct option, without any additional text.\n\n
    Question: {question}\n\n
    Choices:\n
    a. {o1}\n
    b. {o2}\n
    c. {o3}\n
    d. {o4}\n
    e. {o5}\n\n
    Answer:
"""

PROMPT_MED_RAG = """The following is a multiple-choice question about how to manage a patient affected by {condition}. 
    Reply only with the letter of the correct option, without any additional text.\n\n
    Answer the question using the provided context.\n\n
    Context: {context}\n\n
    Question: {question}\n\n
    Choices:\n
    a. {o1}\n
    b. {o2}\n
    c. {o3}\n
    d. {o4}\n
    e. {o5}\n\n
    Answer:
"""