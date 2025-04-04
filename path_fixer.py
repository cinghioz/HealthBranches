
import os
import pandas as pd
import google.generativeai as genai
import time
import ast
from alive_progress import alive_bar

def gemini_inference(model, chat_session, path, cond, text):
    query = f"The reasoning sequence is as follows: \"{path}\", the context associated is: \"{text}\" \
                and the symptom/condition to be treated is: \"{cond}\"."

    lock = True

    while lock:
        time.sleep(3)
        # chat_session = model.start_chat(history=[])
        response = chat_session.send_message(query)
        if len(response.text.split('->')) > 1:
            lock = False

    return response.text

genai.configure(api_key='AIzaSyD_aI_M2ysuA1AhhQI-WoaTlMMOm0njqbk')
# genai.configure(api_key="AIzaSyC-xkk_sjuGdLOTc-MvrjBI4Bdww5ubo4s") # matr
# genai.configure(api_key="AIzaSyAOTNKpnJtD5XVjipFAaEYjxm-ZkYEa_74") # pucci

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
  system_instruction='''Given a sequence of reasoning and associated context, you have to refine it so 
                        that it is uniform with medical terminology. The context is a sequence of sentences that can 
                        be used to clarify the reasoning.\n
                        The meaning of the sequence must NOT change (at most, you can remove superfluous information). 
                        If the reasoning step is poorly explained or ambiguous, refine it using the context and your medical knowledge.\n
                        The answer must be a sequence of reasoning with -> indicating the transition between one step and the next. 
                        Do not add any more text or reasoning in your answer, just the sequence.\n
                        '''
)

chat_session = model.start_chat(history=[])

txt_folder_name = "/Users/cinghio/Documents/PHD/HealthBranches/data/kgbase"
dataset = pd.read_csv("/Users/cinghio/Documents/PHD/HealthBranches/questions_pro/dataset_updated.csv", sep=",", encoding='utf-8')

csv_path = "/Users/cinghio/Documents/PHD/HealthBranches/refined_paths.csv"
start_index = -1  

if os.path.exists(csv_path):
    new_paths = pd.read_csv(csv_path, sep=",", encoding='utf-8')

    if not new_paths.empty:
        start_index = new_paths.tail(1)["index"].values[0]
    print("CSV loaded successfully.")
else:
    new_paths = pd.DataFrame()

results = []

with alive_bar(len(dataset)) as bar:
    for index, data in dataset.iterrows():

        if index <= start_index:
            continue

        condition = data['condition']
        
        try:
            with open(os.path.join(txt_folder_name, condition.upper() + '.txt'), 'r') as file:
                text = file.readlines()
        except Exception:
            print(f"{condition.upper()} text is EMPTY!")
            continue
        
        output = gemini_inference(model, chat_session, data['path'], condition, text)

        # print(f"Initial path: {data['path']}")
        # print(f"Processed path: {output}")

        # Append current iteration result to the list
        results.append({
            'index': index,
            'old_path': data['path'],
            'new_path': output,
            'condition': condition
        })
        
        # Every 5 iterations, merge new results with existing CSV and save
        if index % 50 == 0:
            combined = pd.concat([new_paths, pd.DataFrame(results)], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")

            new_paths = combined.copy()
            results = [] 
        
        bar()

# If there are any remaining results not saved (if not on a 5-iteration boundary)
if results:
    combined = pd.concat([new_paths, pd.DataFrame(results)], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"Saved final CSV file: {csv_path}")


