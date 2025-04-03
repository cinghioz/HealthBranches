from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

class LLMinference:
    def __init__(self, llm_name: str, temperature: float = 0.01, num_predict: int = 128):
        self.llm_name = llm_name
        self.model = OllamaLLM(model=llm_name, temperature=temperature, num_predict=num_predict) 

    def _transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

    def _remove_reasoning(self, text):
        """
        Remove any chain-of-thought reasoning enclosed in <think>...</think> tags.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def single_inference(self, query: str, template: str, path: str, text: str,  choices: List[str], cond: str,  context) -> str | List[str]:
        context_text = "\n\n---\n\n".join([doc.page_content for doc in context])
        prompt_template = ChatPromptTemplate.from_template(template)
        
        if choices: # quiz
            if path != "" and text != "":
                prompt = prompt_template.format(context=context_text, question=query, path=path, text=text, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
            else:
                prompt = prompt_template.format(context=context_text, question=query, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
        else: # open question
            if path != "" and text != "":
                prompt = prompt_template.format(context=context_text, question=query, path=path, text=text, condition=cond)
            else:
                prompt = prompt_template.format(context=context_text, question=query, condition=cond)

        if "deepseek" in self.llm_name.lower():
            instruction = (
                "Answer the question directly with the correct option only (e.g., A, B, C, D, or E). "
                "Do NOT include any internal reasoning or chain-of-thought in your response.\n\n"
            )
            prompt = instruction + prompt

        response_text = self.model.invoke(prompt)
        response_text = response_text.strip().replace("\n", "").replace("  ", "")

        if "deepseek" in self.llm_name.lower():
            response_text = self._remove_reasoning(response_text)

        sources = [doc.metadata.get("source", None) for doc in context]
        
        return response_text, sources

    def qea_evaluation(self, query: str, template: str, path: str, txt: str, choices: List[str], cond: str, vector_store, k: int = 3) -> str:

        results = vector_store.search(query=query, k=3)

        response, sources = self.single_inference(query, template, path, txt, choices, cond, results)

        return response

class LLMinference_deep:
    def __init__(self, llm_name: str, temperature: float = 0.01, num_predict: int = 128, device: int = 0):
        self.llm_name = "deepseek-ai/"+llm_name
        self.temperature = temperature
        self.num_predict = num_predict
        # Load tokenizer and model in FP16 (if using GPU; for CPU set device=-1 and remove torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        # Create a text-generation pipeline; set device=0 for GPU (or -1 for CPU)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device=device
        )

    # def _remove_reasoning(self, text):
    #     """
    #     Remove any chain-of-thought reasoning enclosed in <think>...</think> tags.
    #     """
    #     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _remove_reasoning(self, text):
        """
        Remove all text before the closing </think> tag.
        If the tag is found, returns the text after the tag; otherwise, returns the original text.
        """
        pos = text.find("</think>")
        if pos != -1:
            return text[pos + len("</think>"):].strip()
        return text.strip()

    def invoke(self, prompt: str) -> str:
        """
        Invoke the model with the given prompt and return the generated text.
        """
        d = """You are a professional assistant. Answer the question directly and do not include any internal reasoning or chain-of-thought.\n """+prompt
        outputs = self.generator(d, max_new_tokens=self.num_predict, temperature=self.temperature, do_sample=True)
        # Extract generated text from the first output
        # print(outputs)
        for output in outputs:
            answer = output['generated_text']
            final_answer = self._remove_reasoning(answer)
            # print("Final Answer:")
            # print(final_answer)

        response_text = final_answer
        return response_text

    def single_inference(self, query: str, template: str, path: str, text: str, choices: List[str], cond: str, context):
        # Join context documents (assumed to be a list of objects with page_content attribute)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in context])
        prompt_template = ChatPromptTemplate.from_template(template)
        
        if choices:  # quiz mode
            if path != "" and text != "":
                prompt = prompt_template.format(
                    context=context_text,
                    question=query,
                    path=path,
                    text=text,
                    condition=cond,
                    o1=choices[0],
                    o2=choices[1],
                    o3=choices[2],
                    o4=choices[3],
                    o5=choices[4]
                )
            else:
                prompt = prompt_template.format(
                    context=context_text,
                    question=query,
                    condition=cond,
                    o1=choices[0],
                    o2=choices[1],
                    o3=choices[2],
                    o4=choices[3],
                    o5=choices[4]
                )
        else:  # open question
            if path != "" and text != "":
                prompt = prompt_template.format(
                    context=context_text,
                    question=query,
                    path=path,
                    text=text,
                    condition=cond
                )
            else:
                prompt = prompt_template.format(
                    context=context_text,
                    question=query,
                    condition=cond
                )

        response_text = self.invoke(prompt)
        response_text = response_text.strip().replace("\n", "").replace("  ", "")
        sources = [doc.metadata.get("source", None) for doc in context]
        
        return response_text, sources

    def qea_evaluation(self, query: str, template: str, path: str, txt: str, choices: List[str], cond: str, vector_store, k: int = 3) -> str:
        results = vector_store.search(query=query, k=k)
        response, sources = self.single_inference(query, template, path, txt, choices, cond, results)
        return response
