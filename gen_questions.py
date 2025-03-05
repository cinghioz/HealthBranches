import os
import pandas as pd
import google.generativeai as genai
import time
import ast

def gemini_inference(model, path, text, cond):
    query = f"The reasoning sequence is as follows: \"{path}\", the context associated is: \"{text}\" and the symptom/condition to be treated is: \"{cond}\"."
    lock = False

    while not lock:
        time.sleep(7)
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(query)
        to_list = ast.literal_eval(response.text)
        lock = parse_output(to_list)

    to_list.append(path)
    to_list.append(cond)

    return to_list

def parse_output(question_data):
    if not isinstance(question_data, list) or len(question_data) != 4:
        print("The main list must have exactly four elements.")
        return False

    question, answer, options, correct_option = question_data

    if not isinstance(question, str) or not isinstance(answer, str):
        print("The main list must have exactly four elements.")
        return False

    try:
        options = options.replace("['", '["').replace("']", '"]').replace("', '", '", "')
        options = ast.literal_eval(options)
    except (SyntaxError, ValueError):
        print("The third element must be a valid list in string format.")
        return False

    if not isinstance(options, list) or len(options) != 5 or not all(isinstance(opt, str) for opt in options):
        print("The options list must contain exactly five string elements.")
        return False

    if not isinstance(correct_option, str) or len(correct_option) != 1 or correct_option not in "ABCDEabcde":
        print("The correct option must be a single letter (A-E)")
        return False
    
    if (answer.lower() not in options[ord(correct_option.upper()) - 65].lower()) or (options[ord(correct_option.upper()) - 65].lower() not in answer.lower()):
        print("The correct option must be present in the answer.")   
        # print(f"Correct option: {options[ord(correct_option.upper()) - 65].lower()}\n")
        # print(f"Answer: {answer.lower()}\n")     
        return False
    return True

genai.configure(api_key='AIzaSyD_aI_M2ysuA1AhhQI-WoaTlMMOm0njqbk')
# genai.configure(api_key="AIzaSyC-xkk_sjuGdLOTc-MvrjBI4Bdww5ubo4s") # matr
# genai.configure(api_key="AIzaSyAOTNKpnJtD5XVjipFAaEYjxm-ZkYEa_74") # pucci

# Create the model
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8196,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  # model_name="gemini-2.0-pro-exp-02-05",
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  system_instruction='''Given a sequence of reasoning and a text related to it about how to treat a symptom/condition, generate:\n\n
                        1. a question reflecting the reasoning of the sequence provided. The question must also include a clinical case, e.g: "A 67-year-old man is brought to the physician 
                          because of increasing forgetfulness, unsteadiness, and falls over the past year..."\n
                        2. A set of 5 possible answers (A,B,C,D,E). Should not be too long and should also reflect the sequence of reasoning. One of them must be the correct answer. 
                          The other answers need not be correct for the generated question but must be related to the topic of the question.\n\n 
                        The sequence does NOT have to be explicit in both question and answers!\n\n
                        The correct option must be the same as the answer.\n\n
                        The output should be structured in the following format:\n
                        ["question", "answer", "['Option A', 'Option B', 'Option C', 'Option D', 'Option E']", "letter of correct option"]\n\n
                        Do not generate any additional texts.\n\n
                        '''
)

path_folder_name = '/home/cc/PHD/ragkg/paths/'
txt_folder_name = "/home/cc/PHD/ragkg/data/kgbase-new/"
csv_paths = os.listdir(path_folder_name)

ultimate_questions_path = "/home/cc/PHD/ragkg/questions_pro/ultimate_questions_v2.csv"

# Load ultimate_questions CSV
try:
    ultimate_questions = pd.read_csv(ultimate_questions_path, sep=",", encoding='utf-8')
    processed_conditions = set(ultimate_questions["condition"].astype(str).unique())
except FileNotFoundError:
    ultimate_questions = pd.DataFrame(columns=['question', 'answer', 'options', 'correct_option', 'path', 'condition'])
    processed_conditions = set()

# Extract unique "cond" values from csv_paths
csv_names = {csv.split(".")[0].lower() for csv in csv_paths}

# Filter out already processed CSVs
csvs_to_process = [csv for csv in csv_paths if csv.split(".")[0].lower() not in processed_conditions]

qeas = []

print(f"Remaining CSVs to process: {len(csvs_to_process)}")
for csv in csvs_to_process:
    try:
        paths = pd.read_csv(os.path.join(path_folder_name, csv), sep=",")
    except Exception:
        print(f"{csv} path is EMPTY!")
        continue
    
    try:
        with open(os.path.join(txt_folder_name, csv.replace("csv", "txt")), 'r') as file:
            text = file.readlines()
    except Exception:
        print(os.path.join(txt_folder_name, csv))
        print(f"{csv} text is EMPTY!")
        continue

    cond = csv.split(".")[0].lower()
    
    for _, row in paths.iterrows():
        try:
            if '||' in row['paths']:
                sub_paths = row['paths'].split('||')[:2] if len(row['paths'].split('||')) > 2 else row['paths'].split('||')
                qeas.extend([gemini_inference(model, path, text, cond) for path in sub_paths])
            else:
                qeas.append(gemini_inference(model, row['paths'], text, cond))
        except Exception:
            print("MODEL IN ERROR, SKIP ROW...")
            continue

    # Create DataFrame and append new rows
    new_df = pd.DataFrame(qeas, columns=['question', 'answer', 'options', 'correct_option', 'path', 'condition'])
    updated_df = pd.concat([ultimate_questions, new_df], ignore_index=True)

    # Save updated CSV
    updated_df.to_csv(ultimate_questions_path, index=False, encoding='utf-8')

    print(f"{csv} processed!")

print("END!")