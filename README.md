<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/cinghioz/">
    <img src="images/HB.png" alt="Logo" width="500" height="450">
  </a>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

HealthBranches is a new medical Q&A benchmark for complex, multi-step reasoning. It contains 4,063 realistic patient cases across 17 topics, each derived from validated decision pathways in medical texts. Questions come in both open-ended and multiple-choice formats, and every example includes its full clinical reasoning chain. HealthBranches offers a transparent, high-stakes evaluation resource and can also support medical education.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* <p align="left">Python3</p>
* <p align="left">Ollama</p>
* <p align="left">1 ore more gemini keys</p>
* <p align="left">(suggested) GPU with 16GB memory dedicated or higer</p>

### Installation

1. <p align="left">Get a free API Key at [https://aistudio.google.com/apikey]</p>
2. <p align="left">Install Ollama [https://ollama.com/download]</p>
3. <p align="left">Install python libraries</p>

   ```sh
   pip3 install -r requiriments.txt
   ```
4. <p align="left">Install models</p>

   ```sh
   sh pull-models.sh
   ```
5. <p align="left">Add 1 ore more keys in api_keys.txt (1 key x row)</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Data preparation

First of all, the data with which the questions will be created must be added. Two types of data are needed: the text associated with a condition/symptom (.txt) and the associated paths (.csv):
1. <p align="left">Texts must be added in the data/kgbase folder, one for each condition</p>
2. <p align="left">Paths must be added in the paths folder, one for each condition</p>. Each line of a csv containing paths consists of 3 elements: <b>source</b>, <b>leaf</b>, <b>paths</b>. The source will always be the same (the problem to be solved) while the leaves change according to the sequence of decisions made in the path. If there are more thant one path going from the root to the same leaf, they are entered in the path field separated by the string ‘||’. A path is a string containing a string for each node in the path describing it, and ‘->’ indicating the transition between one node and the next (<i>see csvs in the paths folder for more information</i>).

### Path refinement (optional)

This step standardize path nodes with medical terminology and resolves ambiguities. 
   ```python
   python3 path_fixer.py
   ```

### Q&A generation

### LLMs benchmarking

### Evaluate results

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
