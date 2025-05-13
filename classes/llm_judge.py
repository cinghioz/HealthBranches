import google.generativeai as genai
import pandas as pd
import os
import time
import argparse
import logging
from io import StringIO
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', filename='llm_judge.log')

# --- Gemini API Prompt ---
# Define the detailed prompt for the Gemini API
EVALUATION_PROMPT_TEMPLATE = """
*** TASK INTRODUCTION:
You are an evaluator for Q&A medical tasks. You will evaluate the quality of answers to medical questions. You will have some ground truth to help you evaluate.

You will be given inputs in csv format as a batch of samples. The columns will contain:
- question_id
- question
- reasoning path
- real answer
- predicted answer

The output must be a csv formatted as one line per sample and two columns: 'question_id' and 'score'. Provide ONLY the CSV content in your response, without any introductory text, explanations, or backticks.

*** EVALUATION CRITERIA:
The evaluation criteria must take into account how much the given answer (predicted answer) is adherent to the reasoning path and to the ground truth answer (real answer).
The evaluation MUST not take into account the "style" of the answer but only the content.
The score must be an integer between 0 to 10.

Preparation Phase (Per Batch):
Understand the Task: Re-read the task description and evaluation criteria carefully. Confirm the inputs (Question, Reasoning Path, Real Answer, Predicted Answer) and the required output (CSV with 'question_id' and 'score' columns, 0-10 integer score). Internalize the core evaluation principle: content adherence to both the reasoning path and the ground truth answer, ignoring style.
Reasoning: Ensures the evaluator is aligned with the task requirements before starting the evaluation of individual samples.

Evaluation Phase (Per Sample):
Step 1: Contextual Understanding (Input Analysis): For the current sample, read and fully comprehend:
The Question: Understand what is being asked.
The Reasoning Path: Identify the key steps, logic, evidence, or concepts that should lead to the correct answer according to the ground truth. Note down the crucial elements mentioned.
The Real Answer (Ground Truth): Understand the expected final conclusion, facts, or information that constitutes a correct answer.
Reasoning: This step establishes the 'gold standard' for both the logical process (Reasoning Path) and the final outcome (Real Answer). Without fully grasping these, a fair comparison is impossible.
Step 2: Predicted Answer Analysis: Read and fully comprehend the Predicted Answer provided for the current sample. Identify its main points, conclusions, and any supporting logic or facts it presents.
Reasoning: Isolates the content of the answer being evaluated, preparing it for comparison against the ground truth components.
Step 3: Evaluate Adherence to Reasoning Path: Compare the content of the Predicted Answer (from Step 2) against the key elements of the Reasoning Path (identified in Step 1).
Action: Assess:
Does the Predicted Answer reflect the core logic or sequence described in the Reasoning Path?
Does it incorporate key terms, concepts, or intermediate steps mentioned in the Reasoning Path?
Does it contradict or ignore significant parts of the Reasoning Path?
Focus: Look for conceptual alignment and inclusion of critical reasoning components, not verbatim matching or stylistic similarity.
Reasoning: This directly addresses the first part of the evaluation criteria – "how much the given answer... is adherent to the reasoning path". It checks if the process implied by the predicted answer aligns with the expected process.
Step 4: Evaluate Adherence to Ground Truth Answer: Compare the content of the Predicted Answer (from Step 2) against the Real Answer (Ground Truth) (identified in Step 1).
Action: Assess:
Does the Predicted Answer convey the same core medical information, diagnosis, recommendation, or conclusion as the Real Answer?
Does it miss critical facts present in the Real Answer?
Does it introduce factually incorrect or contradictory information compared to the Real Answer?
Focus: Look for factual accuracy and completeness of the final information, ignoring phrasing, sentence structure, or length unless it fundamentally changes the meaning.
Reasoning: This directly addresses the second part of the evaluation criteria – "how much the given answer... is adherent... to the ground truth answer". It checks the correctness and completeness of the outcome.
Step 5: Synthesize Findings & Assign Provisional Score (0-10 Scale): Based on the assessments in Step 3 and Step 4, determine an overall adherence score. Consider the relative importance and severity of any deviations.
Action: Mentally map the adherence levels to the 0-10 scale:
10: Perfect or near-perfect alignment with both the Reasoning Path's core logic and the Real Answer's core content. Minor stylistic variations are ignored.
8-9: Strong alignment with both, perhaps missing a minor nuance from the reasoning or phrasing the final answer slightly less completely, but fundamentally correct and well-reasoned.
6-7: Good alignment with the Real Answer, but shows noticeable (though not critical) deviations from the Reasoning Path, OR strong adherence to the Reasoning Path but minor inaccuracies/omissions in the final answer compared to the Real Answer. The core message is largely correct.
4-5: Moderate alignment. May follow parts of the reasoning but miss key steps, OR get the final answer partially right but miss significant elements or include some inaccuracies. Alternatively, may align well with one (e.g., Real Answer) but completely ignore the other (Reasoning Path).
2-3: Poor alignment. Significant divergence from the Reasoning Path AND/OR major inaccuracies or omissions compared to the Real Answer. Contains elements of relevance but is fundamentally flawed.
1: Very poor alignment. Barely addresses the question or reasoning, contains mostly incorrect information relevant to the Real Answer.
0: Completely incorrect, irrelevant, off-topic, or nonsensical. No meaningful alignment with either Reasoning Path or Real Answer.
Reasoning: This step translates the qualitative comparison into a quantitative score reflecting the overall content quality based on the dual criteria (Reasoning Path + Real Answer adherence). The scale provides granular levels to reflect different degrees of quality.
Step 6: Final Score Validation: Ensure the assigned score is an integer between 0 and 10.
Action: If the provisional score is not an integer (e.g., mentally assigned 7.5), round it to the nearest integer (e.g., 8). Double-check it falls within the 0-10 range.
Reasoning: Adheres to the specific output format requirement for the score (integer, 0-10).

Post-Evaluation Phase (Per Batch):
Step 7: Repeat for All Samples: Execute Steps 1 through 6 for each sample row in the input batch CSV.
Reasoning: Ensures every provided sample is evaluated consistently using the defined process.
Step 8: Format Output CSV: Compile the results into the final output format.
Action: Create a new CSV file (or modify the input). Ensure it contains ONLY the 'question_id' and 'score' columns for the evaluated samples.
Reasoning: Fulfills the precise output formatting requirements specified in the task description.

*** INPUT BATCH (CSV format):
{batch_csv_data}

*** OUTPUT (CSV format, only 'question_id' and 'score' columns):
"""

# --- API Key Manager ---
class ApiKeyManager:
    """Manages a list of API keys and rotates them."""
    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        self.api_keys = api_keys
        self.current_key_index = 0

    def get_key(self):
        """Returns the current API key."""
        return self.api_keys[self.current_key_index]

    def next_key(self):
        """Moves to the next API key, returns False if all keys have been tried."""
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            self.current_key_index = 0 # Reset to try again from the start if needed
            logging.warning("Rotated through all API keys.")
            return False # Indicate that all keys have been tried in this cycle
        logging.info(f"Switching to API key index {self.current_key_index}")
        return True # Indicate successful rotation

# --- Gemini API Call Function ---
def get_gemini_evaluation(batch_df, api_key_manager, model_name="gemini-1.5-flash", max_retries=3):
    """
    Sends a batch of data to the Gemini API for evaluation.

    Args:
        batch_df (pd.DataFrame): DataFrame containing the batch data.
        api_key_manager (ApiKeyManager): The API key manager instance.
        model_name (str): The name of the Gemini model to use.
        max_retries (int): Maximum number of retries per API key before rotating.

    Returns:
        pd.DataFrame: DataFrame with 'question_id' and 'score', or None if evaluation fails.
    """
    # Ensure required columns are present
    # Standardize column names for checking (lowercase, replace spaces with underscores)
    batch_df.columns = batch_df.columns.str.lower().str.replace(' ', '_')
    required_cols_std = ['question_id', 'question', 'reasoning_path', 'real_answer', 'predicted_answer']
    required_cols_orig = ['question_id', 'question', 'reasoning path', 'real answer', 'predicted answer'] # Keep original for prompt if needed

    # Check using standardized names
    if not all(col in batch_df.columns for col in required_cols_std):
        missing = [col for col in required_cols_std if col not in batch_df.columns]
        logging.error(f"Input DataFrame is missing required columns. Found: {batch_df.columns.tolist()}. Required (standardized): {required_cols_std}. Missing: {missing}")
        return None

    # Prepare the input CSV string for the prompt using original-like names if possible
    # Create a temporary df with the required original names for the prompt CSV
    prompt_df = pd.DataFrame()
    try:
        prompt_df['question_id'] = batch_df['question_id']
        prompt_df['question'] = batch_df['question']
        prompt_df['reasoning path'] = batch_df['reasoning_path']
        prompt_df['real answer'] = batch_df['real_answer']
        prompt_df['predicted answer'] = batch_df['predicted_answer']
    except KeyError as e:
         logging.error(f"Column mapping error when preparing prompt data: {e}. Check input CSV column names.")
         return None

    batch_csv_data = prompt_df.to_csv(index=False)
    prompt = EVALUATION_PROMPT_TEMPLATE.format(batch_csv_data=batch_csv_data)

    initial_key_index = api_key_manager.current_key_index
    keys_tried_in_cycle = 0

    while keys_tried_in_cycle < len(api_key_manager.api_keys):
        api_key = api_key_manager.get_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        retries = 0
        while retries < max_retries:
            try:
                logging.info(f"Sending batch (size: {len(batch_df)}) to Gemini using key index {api_key_manager.current_key_index} (Retry {retries + 1}/{max_retries})...")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1 # Lower temperature for more deterministic output
                    ),
                    safety_settings={
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )

                # --- Response Parsing and Validation ---
                response_text = response.text.strip()
                logging.debug(f"Raw Gemini Response:\n{response_text}")

                # Attempt to parse the response as CSV
                try:
                    # Specify question_id as string during parsing
                    result_df = pd.read_csv(StringIO(response_text), dtype={'question_id': str})
                except Exception as parse_error:
                    logging.error(f"Failed to parse Gemini response as CSV: {parse_error}")
                    logging.error(f"Problematic response text:\n{response_text}")
                    retries += 1
                    time.sleep(2 ** retries)
                    continue

                # Validate columns
                if 'question_id' not in result_df.columns or 'score' not in result_df.columns:
                    logging.error(f"Gemini response missing required columns ('question_id', 'score'). Found: {result_df.columns}")
                    logging.error(f"Problematic response text:\n{response_text}")
                    retries += 1
                    time.sleep(2 ** retries)
                    continue

                # Validate score type and range
                try:
                    result_df['score'] = pd.to_numeric(result_df['score'], errors='coerce')
                    result_df.dropna(subset=['score'], inplace=True)
                    result_df['score'] = result_df['score'].astype(int)
                    invalid_scores = result_df[(result_df['score'] < 0) | (result_df['score'] > 10)]
                    if not invalid_scores.empty:
                        logging.warning(f"Found scores outside the 0-10 range: {invalid_scores.to_dict('records')}")
                        # Clamping scores to be within 0-10 range
                        result_df['score'] = result_df['score'].clip(0, 10)
                        logging.warning("Clamped out-of-range scores to 0-10.")


                except Exception as e:
                    logging.error(f"Error validating/converting scores: {e}")
                    retries += 1
                    time.sleep(2 ** retries)
                    continue

                # Ensure question_id type is string (already done by read_csv, but double check doesn't hurt)
                result_df['question_id'] = result_df['question_id'].astype(str)

                logging.info(f"Successfully received and parsed evaluation for {len(result_df)} items.")
                return result_df[['question_id', 'score']]

            except genai.types.generation_types.BlockedPromptException as bpe:
                 logging.error(f"API call blocked due to safety settings or prompt issues: {bpe}")
                 return None # Fail the batch
            except Exception as e:
                logging.warning(f"API call failed with key index {api_key_manager.current_key_index} (Retry {retries + 1}/{max_retries}): {e}")
                # Consider checking for specific rate limit errors here
                retries += 1
                if retries < max_retries:
                    time.sleep(2 ** retries)
                else:
                    logging.warning(f"Max retries reached for key index {api_key_manager.current_key_index}.")
                    break # Stop retrying with this key

        # If loop finished due to max_retries or specific error, try the next key
        keys_tried_in_cycle += 1
        if not api_key_manager.next_key() and keys_tried_in_cycle < len(api_key_manager.api_keys):
             logging.error("Failed to switch to the next key unexpectedly.")
             break
        elif keys_tried_in_cycle >= len(api_key_manager.api_keys):
             logging.error("All API keys failed for the current batch.")
             break
        time.sleep(1) # Small delay before trying next key

    logging.error(f"Evaluation failed for batch after trying all keys.")
    return None

# --- Helper Functions ---
def find_csv_files(folder_path):
    """Finds all CSV files in the specified folder."""
    csv_files = []
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".csv") and "OPEN" in filename:
                csv_files.append(os.path.join(folder_path, filename))
    except FileNotFoundError:
        logging.error(f"Input folder not found: {folder_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading input folder {folder_path}: {e}")
        return []
    logging.info(f"Found {len(csv_files)} CSV files in {folder_path}.")
    return csv_files

def load_processed_ids(output_file):
    """Loads already processed (question_id, model) pairs from the output file."""
    processed_ids = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0: # Check size > 0
        try:
            # Read output file specifying question_id as string immediately
            df_processed = pd.read_csv(output_file, dtype={'question_id': str})

            # Ensure columns exist before creating tuples
            if 'question_id' in df_processed.columns and 'model' in df_processed.columns:
                # Ensure model column is also string
                df_processed['model'] = df_processed['model'].astype(str)
                # Handle potential NaN/None values before adding to set
                df_processed.dropna(subset=['question_id', 'model'], inplace=True)
                for _, row in df_processed.iterrows():
                    processed_ids.add((row['question_id'], row['model']))
                logging.info(f"Loaded {len(processed_ids)} already processed entries from {output_file}.")
            else:
                logging.warning(f"Output file {output_file} exists but missing 'question_id' or 'model' column. Starting fresh.")
        except pd.errors.EmptyDataError:
            logging.info(f"Output file {output_file} is empty. Starting fresh.")
        except Exception as e:
            logging.error(f"Error reading processed IDs from {output_file}: {e}. Starting fresh.")
    else:
        logging.info(f"Output file {output_file} not found or is empty. Starting fresh.")
    return processed_ids

def get_keys(file_path: str = "api_keys.txt"):
    try:
        with open(file_path, 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
        if not keys:
            print(f"Error: No API keys found in {file_path}")
            return []
        return keys
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please create this file and put your API keys in it, one per line.")
        return []

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate Q&A answers using Gemini API.")
    parser.add_argument("--input_folder", type=str, default="results", help="Path to the folder containing input CSV files.")
    parser.add_argument("--output_file", type=str, default="open_eval/judge_res.csv", help="Path to the output CSV file for results.")
    parser.add_argument("--model_name", default="gemini-2.0-flash", help="Name of the Gemini model to use (e.g., 'gemini-1.5-flash', 'gemini-pro').")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of rows to send to the API in each batch.")
    parser.add_argument("--cooldown", type=float, default=4.5, help="Cooldown period (in seconds) between API calls.")
    parser.add_argument("--max_retries_per_key", type=int, default=3, help="Max retries per API key before switching.")
    parser.add_argument('--pred_col', default='zero_shot', help="Column name for the predicted answer in the input CSV.")


    args = parser.parse_args()

    api_keys = get_keys()
    if not api_keys:
        logging.error("No API keys provided. Please use the --api_keys argument.")
        sys.exit(1)

    key_manager = ApiKeyManager(api_keys)

    processed_ids = load_processed_ids(args.output_file)
    input_files = find_csv_files(args.input_folder)

    if not input_files:
        logging.warning("No input CSV files found. Exiting.")
        sys.exit(0)

    total_processed_count = 0
    total_skipped_count = 0
    total_failed_count = 0

    # Prepare output file header if it doesn't exist
    if not os.path.exists(args.output_file) or os.path.getsize(args.output_file) == 0:
        try:
            pd.DataFrame(columns=['question_id', 'score', 'model']).to_csv(args.output_file, index=False)
            logging.info(f"Created output file with header: {args.output_file}")
        except Exception as e:
            logging.error(f"Failed to create output file {args.output_file}: {e}")
            sys.exit(1)

    logging.info("="*30)
    logging.info("Starting evaluation script...")
    logging.info(f"Input folder: {args.input_folder}")
    logging.info(f"Output file: {args.output_file}")

    for file_path in input_files:
        model_name_from_file = os.path.basename(file_path) # Use filename as model identifier
        logging.info(f"--- Processing file: {model_name_from_file} ---")

        try:
            # Read the input CSV, specifying question_id as string
            df_input = pd.read_csv(file_path, dtype={'question_id': int})
             # Standardize column names (lowercase, replace spaces with underscores)
            df_input.columns = df_input.columns.str.lower().str.replace(' ', '_')
            logging.info(f"Read {len(df_input)} rows from {model_name_from_file}. Columns: {df_input.columns.tolist()}")

            # add the column_id
            df_input['question_id'] = df_input.index

            # rename the columns
            # df_input.rename(columns={'question': 'question', 'path': 'reasoning_path', 'real': 'real_answer', 'zero_shot_path': 'predicted_answer'}, inplace=True)
            df_input.rename(columns={'question': 'question', 'path': 'reasoning_path', 'real': 'real_answer', args.pred_col: 'predicted_answer'}, inplace=True)

            # add exp_type in filename
            model_name_from_file = model_name_from_file + '_exp_type_' + args.pred_col

            # Ensure question_id is present (already checked dtype implicitly)
            if 'question_id' not in df_input.columns:
                 logging.error(f"Skipping file {model_name_from_file}: Missing 'question_id' column after standardization.")
                 continue
            # Ensure it's string type after standardization (redundant if read correctly, but safe)
            df_input['question_id'] = df_input['question_id'].astype(str)


        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping empty file: {file_path}")
            continue
        except FileNotFoundError:
             logging.error(f"File not found during processing loop: {file_path}")
             continue
        except Exception as e:
            logging.error(f"Failed to read or preprocess {file_path}: {e}")
            continue # Skip to the next file

        file_processed_count = 0
        file_skipped_count = 0
        file_failed_count = 0

        for i in range(0, len(df_input), args.batch_size):
            batch_df = df_input.iloc[i:i + args.batch_size].copy()

            # Filter out already processed rows within the batch
            original_batch_size = len(batch_df)
            # Create the comparison key using string types
            batch_df['__processed_key'] = batch_df['question_id'].astype(str) + '||' + model_name_from_file
            # Ensure processed_ids uses strings
            processed_keys_str = set(f"{str(pid)}||{str(pmodel)}" for pid, pmodel in processed_ids)

            batch_to_process = batch_df[~batch_df['__processed_key'].isin(processed_keys_str)].drop(columns=['__processed_key'])

            skipped_in_batch = original_batch_size - len(batch_to_process)
            file_skipped_count += skipped_in_batch
            total_skipped_count += skipped_in_batch

            if batch_to_process.empty:
                if skipped_in_batch > 0:
                    logging.info(f"Skipped batch (rows {i+1}-{i+original_batch_size}) as all items were already processed.")
                continue

            logging.info(f"Processing batch (rows {i+1}-{i+original_batch_size}, {len(batch_to_process)} new items) from {model_name_from_file}...")

            # Call Gemini API
            evaluation_results_df = get_gemini_evaluation(
                batch_to_process,
                key_manager,
                model_name=args.model_name,
                max_retries=args.max_retries_per_key
            )

            # Cooldown after every API attempt
            logging.debug(f"Cooldown for {args.cooldown} seconds...")
            time.sleep(args.cooldown)

            # Process results
            if evaluation_results_df is not None and not evaluation_results_df.empty:
                evaluation_results_df['model'] = model_name_from_file
                # Ensure question_id is string before saving
                evaluation_results_df['question_id'] = evaluation_results_df['question_id'].astype(str)
                output_df = evaluation_results_df[['question_id', 'score', 'model']]

                try:
                    output_df.to_csv(args.output_file, mode='a', header=False, index=False)
                    processed_in_batch = len(output_df)
                    file_processed_count += processed_in_batch
                    total_processed_count += processed_in_batch
                    logging.info(f"Successfully processed and saved {processed_in_batch} items for batch.")

                    # Update the set of processed IDs in memory (ensure using strings)
                    for _, row in output_df.iterrows():
                       processed_ids.add((str(row['question_id']), str(row['model'])))

                except Exception as e:
                    logging.error(f"Failed to write results to {args.output_file}: {e}")
                    failed_in_batch = len(batch_to_process)
                    file_failed_count += failed_in_batch
                    total_failed_count += failed_in_batch

            else:
                failed_in_batch = len(batch_to_process)
                file_failed_count += failed_in_batch
                total_failed_count += failed_in_batch
                logging.error(f"Failed to get evaluation for batch (rows {i+1}-{i+original_batch_size}) from {model_name_from_file}.")


        logging.info(f"--- Finished file: {model_name_from_file} ---")
        logging.info(f"    Processed: {file_processed_count}, Skipped: {file_skipped_count}, Failed: {file_failed_count}")


    logging.info("="*30)
    logging.info("Evaluation script finished.")
    logging.info(f"Total items processed: {total_processed_count}")
    logging.info(f"Total items skipped (already done): {total_skipped_count}")
    logging.info(f"Total items failed: {total_failed_count}")
    logging.info(f"Results saved to: {args.output_file}")
    logging.info("="*30)


if __name__ == "__main__":
    main()
