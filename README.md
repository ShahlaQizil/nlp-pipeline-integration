# NLP Pipeline Integration

An end-to-end Natural Language Processing pipeline that combines **tokenization**, **part-of-speech (POS) tagging**, **named entity recognition (NER)**, and **sentiment analysis** into a single reusable Python function.

The project ships with two pipelines:

1. `analyze_text()` — the basic integration using NLTK + spaCy + a default Hugging Face sentiment model.
2. `analyze_text_improved()` — a refined version that addresses the limitations discovered while testing the basic pipeline.

---

## Features

| Stage | Library | Output |
|---|---|---|
| Tokenization | NLTK / spaCy | List of word tokens |
| POS Tagging | NLTK / spaCy | `(token, tag)` pairs |
| Named Entity Recognition | spaCy (`en_core_web_sm`) | Entities + labels (PERSON, ORG, GPE, DATE...) |
| Sentiment Analysis | Hugging Face Transformers | POSITIVE / NEGATIVE (basic) or POSITIVE / NEUTRAL / NEGATIVE (improved) |

### Improvements in `analyze_text_improved()`

- **Unified tokenizer.** spaCy handles tokenization, POS, and NER in a single pass — tokens stay aligned with entity spans.
- **3-class sentiment.** Uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, which can express *NEUTRAL* (the default DistilBERT model cannot).
- **Per-sentence scoring.** Long documents are split via `doc.sents` before being sent to the transformer, keeping inputs under the 512-token limit and surfacing tone shifts.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS/Linux

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the spaCy English model
python -m spacy download en_core_web_sm
```

NLTK resources (`punkt_tab`, `averaged_perceptron_tagger_eng`) are downloaded automatically the first time you run the script.

---

## Usage

### Run the demo

```bash
python nlp_pipeline.py
```

This processes four built-in sample sentences through both pipelines and prints the results to the console.

### Use it in your own code

```python
from nlp_pipeline import analyze_text, analyze_text_improved

result = analyze_text_improved(
    "Serena Williams won Wimbledon in 2016, solidifying her status as "
    "one of the greatest tennis players in history."
)

print(result["entities"])
# [('Serena Williams', 'PERSON'), ('Wimbledon', 'EVENT'), ('2016', 'DATE')]

print(result["sentence_sentiment"])
# [{'sentence': '...', 'label': 'positive', 'score': 0.95}]
```

---

## Example output

```
======================================================================
INPUT TEXT: The new iPhone released by Apple in California was disappointing and overpriced.
======================================================================

[Step 2] Tokenization:
['The', 'new', 'iPhone', 'released', 'by', 'Apple', 'in', 'California', ...]

[Step 3] POS Tagging:
[('The', 'DT'), ('new', 'JJ'), ('iPhone', 'NN'), ('released', 'VBN'), ...]

[Step 4] Named Entities:
  Apple        -> ORG
  California   -> GPE

[Step 5] Sentiment Analysis:
  [{'label': 'NEGATIVE', 'score': 0.9996}]
```

---

## Reflection notes

While testing the basic pipeline, two limitations surfaced:

1. **spaCy's small model mislabels modern tech names.** "SpaceX" was tagged as `PERSON` instead of `ORG`, and "Natural Language Processing" was tagged as `ORG`. Upgrading to `en_core_web_lg` or `en_core_web_trf` improves accuracy.
2. **Binary sentiment forces neutral text into positive.** A factual sentence like *"Elon Musk announced that SpaceX will launch a new rocket"* is scored POSITIVE with ~99% confidence by the default model. The improved pipeline's 3-class model handles this correctly.

Further improvements worth exploring:

- Coreference resolution (e.g. via [`coreferee`](https://github.com/richardpaulhudson/coreferee)) to track entities across sentences.
- Domain-specific sentiment models (FinBERT for finance, BERTweet for social media).
- Aspect-based sentiment analysis to score *which entity* the sentiment refers to.

---

## Project structure

```
.
├── nlp_pipeline.py     # Main script with both pipelines
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## License

MIT
