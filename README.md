# doccano-mini

doccano-mini is a few-shot annotation tool to assist the development of applications with Large language models (LLMs). Once you annotate a few text, you can solve your task (e.g. text classification) with LLMs via [LangChain](https://github.com/hwchase17/langchain).

At this time, the following tasks are supported:

- Text classification
- Question answering
- Summarization
- Paraphrasing
- Named Entity Recognition
- Task Free

Note: This is an experimental project.

## Installation

```bash
pip install doccano-mini
```

## Usage

For this example, we will be using OpenAI’s APIs, so we need to set the environment variable in the terminal.

```bash
export OPENAI_API_KEY="..."
```

Then, we can run the server.

```bash
doccano-mini
```

Now, we can open the browser and go to `http://localhost:8501/` to see the interface.

### Step1: Annotate a few text

In this step, we will annotate a few text. We can add a new text by clicking the `+` button. Try it out by double-clicking on any cell. You'll notice you can edit all cell values.

![Step1](https://raw.githubusercontent.com/doccano/doccano-mini/master/docs/images/annotation.gif)

The editor also supports pasting in tabular data from Google Sheets, Excel, and many other similar tools.

![Copy and Paste](https://raw.githubusercontent.com/doccano/doccano-mini/master/docs/images/copy_and_paste.gif)

### Step2: Test your task

In this step, we will test your task. We can enter a new test to the text box and click the `Predict` button. Then, we can see the result of the test.

<img src="https://raw.githubusercontent.com/doccano/doccano-mini/master/docs/images/test_new_example.jpg" alt= “Step2” width="700">

### Step3: Download the config

In this step, we will download the [LangChain](https://github.com/hwchase17/langchain)'s config. We can click the `Download` button to download it. After loading the config file, we can predict a label for the new text.

```python
from langchain.chains import load_chain

chain = load_chain("chain.yaml")
chain.run("YOUR TEXT")
```

## Development

```bash
poetry install
streamlit run doccano_mini/home.py
```
