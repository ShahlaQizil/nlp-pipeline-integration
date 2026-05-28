# NLP Pipeline Integration

An end-to-end Natural Language Processing pipeline that runs raw text through
four stages — **tokenization, POS tagging, Named Entity Recognition, and
sentiment analysis** — by integrating three popular NLP libraries into a single
workflow: **NLTK**, **spaCy**, and **Hugging Face Transformers**.

The project ships two pipelines: a baseline version and an improved version that
fixes the weaknesses found in the baseline.

## What it does

| Stage | Tool | Output |
|-------|------|--------|
| Tokenization | NLTK / spaCy | Words split into tokens |
| POS Tagging | NLTK / spaCy | Grammatical role of each token (noun, verb, …) |
| Named Entity Recognition | spaCy (`en_core_web_sm`) | People, organizations, locations, dates |
| Sentiment Analysis | Hugging Face Transformers | Positive / Negative (and Neutral in the improved version) |

## Baseline vs. improved pipeline

The improved pipeline (`analyze_text_improved`) addresses three real problems:

1. **Token alignment** — spaCy handles both tokenization *and* POS tagging, so
   tokens line up with NER spans (the baseline mixed NLTK tokens with spaCy
   entities).
2. **3-class sentiment** — uses `cardiffnlp/twitter-roberta-base-sentiment-latest`
   so genuinely neutral text isn't forced into "positive." Handles informal text well.
3. **Long-document handling** — splits text into sentences before scoring to stay
   under the 512-token transformer limit and to surface per-sentence tone shifts.

## Installation

```bash
pip install nltk spacy transformers torch
python -m spacy download en_core_web_sm
```

## Usage

```bash
python nlp_pipeline.py
```

This runs both pipelines over a set of sample sentences and prints each stage's
output. To analyze your own text:

```python
from nlp_pipeline import analyze_text_improved

analyze_text_improved("Apple released a new iPhone in California last month.")
```

## Example output

```
[Step 4] Named Entities:
  Apple                     -> ORG
  California                -> GPE
  last month                -> DATE

[Step 5] Sentiment Analysis (3-class, per sentence):
  [NEUTRAL  0.812] Apple released a new iPhone in California last month.
```

## Tech stack
Python · NLTK · spaCy · Hugging Face Transformers · PyTorch
