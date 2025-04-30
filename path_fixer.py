import os
import pandas as pd
import google.generativeai as genai
import time
import ast
import sys
from alive_progress import alive_bar

def load_api_keys(filename="api_keys.txt"):
    """Loads API keys from a file.

    Args:
        filename (str, optional): The name of the file containing API keys. Defaults to "api_keys.txt".
    Returns:
        list: A list of API keys.  Returns an empty list if the file is not found or empty.
    """
    try:
        with open(filename, 'r') as f:
            keys = [line.strip() for line in f if line.strip()]  # Read and remove empty lines
        if not keys:
            print(f"Error: No API keys found in {filename}")
            return []
        return keys
    except FileNotFoundError:
        print(f"Error: {filename} not found.  Please create this file and put your API keys in it, one per line.")
        return []

def gemini_inference(model, path, cond, text, api_keys, key_index):
    """Performs inference using the Gemini model, with retry logic for API key failures.

    Args:
        model: The Gemini GenerativeModel instance.
        path (str): The path for the reasoning sequence.
        cond (str): The symptom/condition to be treated.
        text (list): The context associated with the reasoning sequence.
        api_keys (list): A list of API keys.
        key_index (int): The index of the current API key to use.

    Returns:
        tuple: (str, int) The refined path and the updated key_index.  Returns (None, updated_key_index) on failure.
    """
    query = f"The reasoning sequence is as follows: \"{path}\" and the symptom/condition to be treated is: \"{cond}\"."
    max_retries = 5
    retries = 0
    output = path  # Default output

    while retries <= max_retries:
        time.sleep(5)
        genai.configure(api_key=api_keys[key_index])
        chat_session = model.start_chat(history=[])
        try:
            response = chat_session.send_message(query)
            if len(response.text.split('->')) > 1:
                output = response.text
                return output, key_index
            else:
                print(f"Refinement condition not met. Retrying with same key. Retry count: {retries}")
                retries += 1
        except Exception as e:
            print(f"Error during inference: {e}")
            retries += 1
            key_index = (key_index + 1) % len(api_keys)
            print(f"Trying a different API key. New key index: {key_index}, Retry count: {retries}")

        if retries > max_retries:
            print("Max retries exceeded. Skipping this path.")
            return None, key_index  # Return None and updated key_index

    return output, key_index  # Return the last output and updated key_index



def main():
    """Main function to orchestrate the data processing and path refinement."""
    api_keys = load_api_keys()
    if not api_keys:
        print("No API keys available. Exiting.")
        sys.exit(1)

    # Create the model
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8196,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        # model_name="gemini-2.0-pro-exp-02-05",
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        system_instruction='''Given a sequence of reasoning and associated condition, you have to refine it so 
                            that it is uniform with medical terminology.\n
                            The meaning of the sequence must NOT change (at most, you can remove superfluous information or sanitize the text). 
                            If the reasoning step is poorly explained or ambiguous, refine it using the context and your medical knowledge.\n
                            The answer must be a sequence of reasoning with -> indicating the transition between one step and the next. 
                            Do not add any more text or reasoning in your answer, just the sequence.\n
                            '''
    )

    txt_folder_name = "/home/cc/PHD/HealthBranches/data/kgbase"
    dataset_path = "/home/cc/PHD/HealthBranches/questions_pro/dataset_updated.csv"
    csv_path = "/home/cc/PHD/HealthBranches/refined_paths.csv"
    start_index = 0

    # Load dataset CSV
    try:
        dataset = pd.read_csv(dataset_path, sep=",", encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Dataset CSV file not found at {dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading dataset CSV: {e}")
        sys.exit(1)

    # Load or create refined paths CSV
    if os.path.exists(csv_path):
        try:
            new_paths = pd.read_csv(csv_path, sep=",", encoding='utf-8')
            if not new_paths.empty:
                start_index = new_paths.tail(1)["index"].values[0]
            print("Refined paths CSV loaded successfully.")
        except Exception as e:
            print(f"Error reading existing refined paths CSV: {e}. Creating a new one.")
            new_paths = pd.DataFrame()
    else:
        new_paths = pd.DataFrame()

    results = []
    key_index = 0  # Initialize API key index

    if start_index > 0:
        print(f"Processing from index {start_index}...")

    with alive_bar(len(dataset[start_index:])) as bar:
        for index, data in dataset[start_index:].iterrows():
            condition = data['condition']
            try:
                with open(os.path.join(txt_folder_name, condition.upper() + '.txt'), 'r') as file:
                    text = file.readlines()
            except Exception:
                print(f"{condition.upper()} text is EMPTY!")
                continue

            refined_path, key_index = gemini_inference(model, data['path'], condition, text, api_keys, key_index)
            if refined_path is None:
                print(f"Failed to refine path for index {index}, condition {condition}. Skipping.")
                continue  # Skip to the next iteration

            results.append({
                'index': index,
                'question': data['question'],
                'options': data['options'],
                'correct_option': data['correct_option'],
                'old_path': data['path'],
                'new_path': refined_path,
                'condition': condition
            })

            # Keep history within reasonable bounds.  The original code had a potential memory leak.
            if len(model.start_chat().history) > 100:
                model.start_chat().history = model.start_chat().history[50:]

            if (index + 1) % 50 == 0:
                print(f"Saving progress at index {index}")
                combined = pd.concat([new_paths, pd.DataFrame(results)], ignore_index=True)
                combined.to_csv(csv_path, index=False)
                new_paths = combined.copy()
                results = []  # Clear results

            bar()  # Update progress bar

    # Save any remaining results
    if results:
        combined = pd.concat([new_paths, pd.DataFrame(results)], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"Saved final CSV file: {csv_path}")

if __name__ == "__main__":
    main()
