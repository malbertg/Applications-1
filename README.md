
# ReadWise: Your Q&A Generator 📚

## Overview
ReadWise is an application designed to enhance language proficiency through the generation of questions and answers from uploaded text documents. Utilizing the Mistral 7B model via API and the Langchain library, coupled with a Streamlit interface, ReadWise offers a user-friendly platform for both teachers and students to engage in interactive language learning. It processes PDF documents to generate content focusing on comprehension, vocabulary, and grammar.

## Key Features

- **Interactive Q&A Generation:** Automatically generates practice questions and coherent answers to facilitate language acqusition. 🧠
- **PDF Uploads:** Users can upload PDF documents for text extraction and question generation. 📄
- **Custom Tailored Content:** Supports language learning by generating content aligned with the user's learning objectives and preferences. 💡
- **Streamlit Interface:** Provides an easy-to-use web interface for an engaging user experience. 🖥️

![Screenshot 1: Upload Interface](path/to/your/screenshot1.png)
*Screenshot 1: Upload Interface*

![Screenshot 2: Q&A Generation](path/to/your/screenshot2.png)
*Screenshot 2: Q&A Generation*

## Installation

### Prerequisites

- Python 3.8 or later. 🐍

### Recommended Setup

It is highly recommended to use a virtual environment for this project to manage dependencies effectively and isolate the project's execution environment.

1. **Clone the repository:**

```bash
git clone https://github.com/yourgithubusername/readwise-qa.git
cd readwise-qa
```
2. **Create and activate a v##irtual environment:**
For Unix/macOS:
```bash
python3 -m venv env
source env/bin/
```
For Windows:
```bash
python -m venv env
.\env\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Set up your environment variables:**
Create a .env file in the root directory of the project and add your environment variables. For example:
```bash
REPLICATE_API_KEY=your_replicate_api_key_here
```
## Usage

To run ReadWise, execute the following command from the terminal:
```bash
streamlit run your_script_name.py
```
Access the application via the URL that pops up to start using ReadWise. 🚀

###Contributions
Your contributions to ReadWise are welcome! Feel free to fork the repository, make your changes, and submit a pull request. ❤️

###License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
