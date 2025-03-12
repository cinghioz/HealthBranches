from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from typing import List

class LLMinference:
    def __init__(self, llm_name, temperature=0.0, num_predict=128):
        self.llm_name = llm_name
        self.model = OllamaLLM(model=llm_name, temperature=temperature, num_predict=num_predict) 

    def _transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

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

        response_text = self.model.invoke(prompt)
        response_text = response_text.strip().replace("\n", "").replace("  ", "")

        sources = [doc.metadata.get("source", None) for doc in context]
        
        return response_text, sources

    def qea_evaluation(self, query: str, template: str, path: str, txt: str, choices: List[str], cond: str, vector_store, k: int = 3) -> str:

        results = vector_store.search(query=query, k=3)

        response, sources = self.single_inference(query, template, path, txt, choices, cond, results)

        return response