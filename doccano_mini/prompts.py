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
