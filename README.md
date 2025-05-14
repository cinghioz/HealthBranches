<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/cinghioz/">
    <img src="images/HB.png" alt="Logo" width="500" height="450">
  </a>

<!-- ABOUT THE PROJECT -->
## <p align="left">About The Project</p>

<p align="left">HealthBranches is a new medical Q&A benchmark for complex, multi-step reasoning. It contains 4,063 realistic patient cases across 17 topics, each derived from validated decision pathways in medical texts. Questions come in both open-ended and multiple-choice formats, and every example includes its full clinical reasoning chain. HealthBranches offers a transparent, high-stakes evaluation resource and can also support medical education.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## <p align="left">Getting Started</p>

### <p align="left">Prerequisites</p>

* <p align="left">Python3</p>
* <p align="left">Ollama</p>
* <p align="left">1 or more Gemini keys</p>
* <p align="left">GPU with at least 16GB of dedicated memory</p>

### <p align="left">Installation</p>

1. <p align="left">Get a free API Key at [https://aistudio.google.com/apikey]</p>
2. <p align="left">Install Ollama [https://ollama.com/download]</p>
3. <p align="left">Install python libraries</p>

   ```sh
   pip3 install -r requirements.txt
   ```
4. <p align="left">Install models</p>

   ```sh
   sh pull-models.sh
   ```
5. <p align="left">Add 1 or more keys in <b>api_keys.txt</b> (1 key x row)</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## <p align="left">Usage</p>

### <p align="left">Data preparation</p>

<p align="left">First of all, you must provide the data used to generate the questions. Two types of data are needed: the text associated with a condition/symptom (.txt) and the associated paths (.csv):</p>
<p align="left">1. Texts must be added to the data/kgbase folder, one for each condition</p>
<p align="left">2. Paths must be added to the paths folder, one for each condition. Each line of a csv containing paths consists of 3 elements: <b>source</b>, <b>leaf</b>, <b>paths</b>. The source will always be the same (the problem to be solved) while the leaves change according to the sequence of decisions made in the path. If there are more than one path going from the root to the same leaf, they are entered in the path field separated by the string ‘||’. A path is a string containing a string for each node in the path describing it, and ‘->’ indicating the transition between one node and the next (<i>see csvs in the paths folder for more information</i>).</p>

| source      | leaf        | path                                               |
| ----------- | ----------- | -------------------------------------------------- |
| A           | F           | A -> B -> F \|\| A -> B -> D -> F \|\| A -> N -> F |
| A           | G           | A -> B -> E -> K -> G                              |


data
  └── kgbase
        ├── DYSPNEA.txt
        ├── FATIGUE.txt
        ├── HYPERTENSION.txt
        ├── OBESITY.txt
        └── PLEURAL_EFFUSION.txt
└── paths
      ├── DYSPNEA.csv
      ├── FATIGUE.csv
      ├── HYPERTENSION.csv
      ├── OBESITY.csv
      └── PLEURAL_EFFUSION.csv


### <p align="left">Q&A generation</p>
<p align="left">This step generates the csv of questions and answers from text and path pairs:</p>

   ```sh
   python3 gen-questions.py
   ```

### <p align="left">RAG base creation</p>

<p align="left">To initialize and index condition texts:</p>

   ```sh
   python3 init-rag.py -kgbase data/kgbase -chunk_size 500 -overlap 150
   ```

### <p align="left">LLMs benchmarking</p>

<p align="left">Once the questions have been generated, it is possible to test different llm available on ollama on the newly created dataset. It is possible to define models in the <b>model.py</b> file: </p>

   ```python
   MODELS = ["mistral:7b", "gemma:7b", "gemma2:9b", "gemma3:4b", "llama3.1:8b","qwen2.5:7b",
          "phi4:14b", "mistral-nemo:12b", "llama2:7b", "deepseek-r1:8b"]
   ```
<p align="left">For each model, it is possible to generate both the results on the quiz and the open answer.There are two scripts to generate the results:</p>
<p align="left">1) <b>Topline</b>: The path and description of the disease is provided in LLM's context. This serves to understand, given the reasoning in context, the performance:</p>

   ```sh
   sh run-topline.sh
   ```
<p align="left">2) <b>Benchmark</b>: The model is given only the question and, in the case of the quiz, also the possible options:</p>

   ```sh
   sh run-benchmark.sh
   ```
   
### <p align="left">Evaluate results</p>

<p align="left">To evaluate performance in accuracy on quizzes: </p>

   ```sh
   python3 eval-quiz.py -res_dir results
   ```
<p align="left">The quiz evaluation script generate a summary csv with the accuracies (<b>models_accuracy.csv</b>) and a bar plot (<b>plot_models.pdf</b>) </p>


<p align="left">To evaluate performance in open-answer setting: </p>

   ```sh
   python3 classes/scorer.py --files_dir [absolute path of results] --save_file [path to save evaluation]
   ```
   ```sh
   python3 classes/judge.py --input_folder [path of results] --output_file [path to save evaluation] --pred_col [target column names (e.g zero_shot)]
   ```
<p align="left">Both evaluation methods generate a csv containing one line per question (results are not aggregated). </p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

