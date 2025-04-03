from evaluation.quiz_evaluator import evaluate_models, get_conditions

models = ["mistral_7b", "gemma_7b", "gemma2_9b", "gemma3_4b", "llama3.1_8b", "qwen2.5_7b", 
          "phi4_14b", "llama2_7b", "Llama-3.3-70B-Instruct-Turbo-Free"]

print(evaluate_models(models))

res = []

for condition in get_conditions():
    res.append(evaluate_models(models, condition))