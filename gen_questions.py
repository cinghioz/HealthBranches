import os
import pandas as pd
import google.generativeai as genai
import time
import ast
import sys

def load_api_keys(filename="api_keys.txt"):
    """Loads API keys from a file.

    Args:
        filename (str, optional): The name of the file containing API keys. Defaults to "api_keys.txt".
    Returns:
        list: A list of API keys. Returns an empty list if the file is not found or empty.
    """
    try:
        with open(filename, 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
        if not keys:
            print(f"Error: No API keys found in {filename}")
            return []
        return keys
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please create this file and put your API keys in it, one per line.")
        return []

def gemini_inference(model, path, text, cond, api_keys, key_index):
    """Performs inference using the Gemini model, with retry logic for API key failures.

    Args:
        model: The Gemini GenerativeModel instance.
        path (str): The path for the reasoning sequence.
        text (list): The context associated with the reasoning sequence.
        cond (str): The symptom/condition to be treated.
        api_keys (list): A list of API keys.
        key_index (int): The index of the current API key to use.

    Returns:
        tuple: (result, key_index)  The result from the Gemini model, or None if all API keys fail, and the updated key_index.
    """
    query = f"The reasoning sequence is as follows: \"{path}\", the context associated is: \"{text}\" and the symptom/condition to be treated is: \"{cond}\"."
    # Limit the number of retries to prevent infinite loops
    max_retries = 10 
    retries = 0

    while retries <= max_retries:
        time.sleep(1)
        genai.configure(api_key=api_keys[key_index])
        chat_session = model.start_chat(history=[])
        try:
            response = chat_session.send_message(query)
            to_list = ast.literal_eval(response.text)
            if parse_output(to_list):
                to_list.append(path)
                to_list.append(cond)
                return to_list, key_index
            else:
                print(f"Output parsing failed. Retrying with the same key. Retry count: {retries}")
                retries += 1

        except Exception as e:
            print(f"Error during inference: {e}")
            retries += 1
            # Cycle to the next key
            key_index = (key_index + 1) % len(api_keys)
            print(f"Trying a different API key. New key index: {key_index}, Retry count: {retries}")
            if retries > max_retries:
                print("Max retries exceeded. Skipping this row.")
                return None, key_index

    return None, key_index



def parse_output(question_data):
    """Parses the output from the model and validates its format and content.

    Args:
        question_data (list): The data returned by the model.

    Returns:
        bool: True if the output is valid, False otherwise.
    """
    if not isinstance(question_data, list) or len(question_data) != 4:
        print("The main list must have exactly four elements.")
        return False

    question, answer, options, correct_option = question_data

    if not isinstance(question, str) or not isinstance(answer, str):
        print("The question and answer must be strings.")
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

    if (answer.lower() not in options[ord(correct_option.upper()) - 65].lower()) or (
            options[ord(correct_option.upper()) - 65].lower() not in answer.lower()):
        print("The correct option must be present in the answer.")
        return False
    return True



def main():
    api_keys = load_api_keys()
    if not api_keys:
        print("No API keys available. Exiting.")
        sys.exit(1)

    # Create model
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8196,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction='''Given a sequence of reasoning and a text related to it about how to treat a symptom/condition, generate:\n
                            1. a question reflecting the reasoning of the sequence provided. The question must also include a clinical case, e.g: "A 67-year-old man is brought to the physician
                              because of increasing forgetfulness, unsteadiness, and falls over the past year..."\n
                            2. A set of 5 possible answers (A,B,C,D,E). Should not be too long and should also reflect the sequence of reasoning. One of them must be the correct answer.
                              The other answers need not be correct for the generated question but must be related to the topic of the question.\n
                            The sequence does NOT have to be explicit in both question and answers!\n
                            The correct option must be the same as the answer.\n
                            The output should be structured in the following format:
                            ["question", "answer", "['Option A', 'Option B', 'Option C', 'Option D', 'Option E']", "letter of correct option"]\n
                            Do not generate any additional texts.
                            '''
    )

    path_folder_name = "paths"
    txt_folder_name = "data/kgbase/"
    csv_paths = os.listdir(path_folder_name)

    ultimate_questions_path = "questions_pro/questions_test.csv"

    # Load the file of already generated questions
    try:
        ultimate_questions = pd.read_csv(ultimate_questions_path, sep=",", encoding='utf-8')
        processed_conditions = set(ultimate_questions["condition"].astype(str).unique())
    except FileNotFoundError:
        ultimate_questions = pd.DataFrame(columns=['question', 'answer', 'options', 'correct_option', 'path', 'condition'])
        processed_conditions = set()

    # Filter out already processed CSV
    csvs_to_process = [csv for csv in csv_paths if csv.split(".")[0].lower() not in processed_conditions]

    qeas = []
    key_index = 0

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
            result = None
            if '||' in row['paths']:
                # Split the paths by '||' and process each sub-path. If there is more than 1 path from the root to the leaf in question, process two (this value can be changed)
                sub_paths = row['paths'].split('||')[:2] if len(row['paths'].split('||')) > 2 else row['paths'].split('||')
                results, key_index = zip(*[gemini_inference(model, path, text, cond, api_keys, key_index) for path in sub_paths])
                results = [r for r in results if r is not None] # Filter out None results
                key_index = key_index[-1] # Update key_index to the last used key

                if results:
                  qeas.extend(results)
                else:
                   print(f"Failed to get results for sub_paths: {sub_paths} in {csv}")

            else:
                result, key_index = gemini_inference(model, row['paths'], text, cond, api_keys, key_index)
                if result:
                  qeas.append(result)
                else:
                  print(f"Failed to get result for path: {row['paths']} in {csv}")
            
        # Create DataFrame and append new rows
        if qeas: 
            new_df = pd.DataFrame(qeas, columns=['question', 'answer', 'options', 'correct_option', 'path', 'condition'])
            updated_df = pd.concat([ultimate_questions, new_df], ignore_index=True)

            # Save updated CSV
            updated_df.to_csv(ultimate_questions_path, index=False, encoding='utf-8')
            ultimate_questions = updated_df
        else:
            print(f"No valid questions generated for {csv}. Skipping save.")

        qeas = []

        print(f"{csv} processed!")

    print("END!")

if __name__ == "__main__":
    main()
