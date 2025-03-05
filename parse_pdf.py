import google.generativeai as genai
import time
import csv
import os
import pandas as pd

def upload_to_gemini(path, mime_type=None):
  time.sleep(10)
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def find_pdf_files(root_folder):
    pdf_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

genai.configure(api_key='AIzaSyD_aI_M2ysuA1AhhQI-WoaTlMMOm0njqbk')
# Create the model
generation_config = {
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  system_instruction="""The uploaded PDF contains information and decisions to be made about a specific disease/condition.\n 
    Each PDF consists of (sorted from the top of the first page): a title, the authors, a brief description, 
    observations (begin with a capital letter followed by a period), a decision graph, 
    a continuation of an observation if it did not fit on the previous page, references.\n
    In some documents there may be tables, ignore them.\n
    """
)

query =  """Analyze the PDF provided and extract the brief description and observations. 
            The title, authors, decision graph and references should NOT be extracted.\n
            There must be ONLY text contained in the provided pdf in the output.\n
          """

root_folder = "/home/cc/PHD/ragkg/Decision Making in Medicine- An Algorithmic Approach"
pdf_paths = find_pdf_files(root_folder)

for pdf_path in pdf_paths:    
    # Get the filename without the extension for the text file name
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_filename = f"{pdf_filename}.txt"
    # Check if the txt file already exists
    if not os.path.exists("/home/cc/PHD/ragkg/data/kgbase-new/"+txt_filename):
        file = upload_to_gemini(pdf_path)
        response = model.generate_content([query, file])

        # Save the response to a txt file
        with open("/home/cc/PHD/ragkg/data/kgbase-new/"+txt_filename, 'w') as txt_file:
            txt_file.write(response.text)
        print(f"Response saved to {txt_filename}")
    else:
        print(f"{txt_filename} already exists. Skipping...")
