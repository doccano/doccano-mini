from doccano_mini.prompts import make_classification_prompt, make_task_free_prompt


def test_make_classification_prompt():
    examples = [
        {"text": "That would be awesome!", "label": "positive"},
        {"text": "This is awful!", "label": "negative"},
    ]

    expected = """\
Classify the text into one of the following labels:
- negative
- positive


text: That would be awesome!
label: positive

text: This is awful!
label: negative

text: It's very hot."""

    input_text = "It's very hot."

    prompt = make_classification_prompt(examples)

    assert prompt.format(input=input_text) == expected


def test_make_task_free_prompt():
    examples = [
        {"English": "I like sushi.", "Japanese": "寿司が好きです。"},
        {"English": "I live in Japan.", "Japanese": "日本に住んでいます。"},
    ]

    expected = """\
English: I like sushi.
Japanese: 寿司が好きです。

English: I live in Japan.
Japanese: 日本に住んでいます。

English: I'm developing doccano-mini."""

    english_text = "I'm developing doccano-mini."

    prompt = make_task_free_prompt(examples)

    assert prompt.format(English=english_text) == expected
