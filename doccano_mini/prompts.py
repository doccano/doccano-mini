import json
from typing import List

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


def make_classification_prompt(examples: List[dict]) -> FewShotPromptTemplate:
    unique_labels = set([example["label"] for example in examples])

    task_instruction = "Classify the text into one of the following labels:\n"
    # Sorting to make label order deterministic
    for label in sorted(unique_labels):
        task_instruction += f"- {label}\n"

    example_prompt = PromptTemplate(input_variables=["text", "label"], template="text: {text}\nlabel: {label}")
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="text: {input}",
        input_variables=["input"],
    )
    return prompt


def make_question_answering_prompt(examples: List[dict]) -> FewShotPromptTemplate:
    task_instruction = (
        "You are a highly intelligent question answering bot. "
        "You take context and question as input and return the answer from the context. "
        "Retain as much information as needed to answer the question at a later time. "
        "If you don't know the answer, you should return N/A."
    )

    example_prompt = PromptTemplate(
        input_variables=["context", "question", "answer"],
        template="context: {context}\nquestion: {question}\nanswer: {answer}",
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="context: {context}\nquestion: {question}",
        input_variables=["context", "question"],
    )
    return prompt


def make_summarization_prompt(examples: List[dict]) -> FewShotPromptTemplate:
    task_instruction = (
        "You are a highly intelligent Summarization system. "
        "You take Passage as input and summarize the passage as an expert."
    )
    example_prompt = PromptTemplate(
        input_variables=["passage", "summary"], template="passage: {passage}\nsummary: {summary}"
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="passage: {passage}",
        input_variables=["passage"],
    )
    return prompt


def make_paraphrase_prompt(examples: List[dict]) -> FewShotPromptTemplate:
    task_instruction = (
        "You are a highly intelligent paraphrasing system. You take text as input and paraphrase it as an expert."
    )
    example_prompt = PromptTemplate(
        input_variables=["text", "paraphrase"], template="text: {text}\nparaphrase: {paraphrase}"
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="text: {text}",
        input_variables=["text"],
    )
    return prompt


def make_task_free_prompt(examples: List[dict]) -> FewShotPromptTemplate:
    columns = list(examples[0])

    task_instruction = f"Predict {columns[-1]} based on {', '.join(columns[:-1])}."
    example_prompt = PromptTemplate(
        input_variables=columns, template="\n".join([f"{column}: {{{column}}}" for column in columns])
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="\n".join([f"{column}: {{{column}}}" for column in columns[:-1]]),
        input_variables=columns[:-1],
    )
    return prompt


def make_named_entity_recognition_prompt(examples: List[dict], **kwargs) -> FewShotPromptTemplate:
    task_instruction = (
        "You are a highly intelligent and accurate Named-entity recognition(NER) system. "
        "You take Passage as input and your task is to recognize and extract specific types of "
        "named entities in that given passage and classify into a set of entity types.\n"
    )
    types = kwargs.get("types", [])
    task_instruction += "The following entity types are allowed:\n"
    for type in types:
        task_instruction += f"- {type}\n"

    for example in examples:
        entities = [
            {"mention": example["text"][entity["start"] : entity["end"]], "type": entity["label"]}
            for entity in example["entities"]
        ]
        example["entities"] = json.dumps(entities)

    example_prompt = PromptTemplate(
        input_variables=["text", "entities"],
        template="text: {text}\nentities: {entities}",
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=task_instruction,
        suffix="text: {{text}}",
        input_variables=["text"],
        template_format="jinja2",
    )
    return prompt
