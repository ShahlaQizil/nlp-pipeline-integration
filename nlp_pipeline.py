# Step 1: Set up the environment
# Install necessary libraries (run once in your terminal):
#   pip install nltk
#   pip install spacy
#   pip install transformers torch
#   python -m spacy download en_core_web_sm

import nltk
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline
# import nltk: Imports the Natural Language Toolkit, a classic library for basic text processing like splitting sentences.
# from nltk.tokenize import word_tokenize: A specific tool to chop a sentence into individual words (tokens).
# import spacy: A modern, fast library used here for "Named Entity Recognition" (finding names, dates, and places).
# from transformers import pipeline: From Hugging Face, this allows us to use a pre-trained Deep Learning model for "Sentiment Analysis" with just one line of code.

# Step 1 (cont.): Download required NLTK resources
# nltk.download: NLTK is modular. These lines download the specific "rulebooks" needed to tokenize words and identify parts of speech (like nouns vs. verbs).
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# Load the pretrained spaCy model for NER (loaded once and reused)
# Loads a small English language model
nlp = spacy.load("en_core_web_sm")

# Initialize the sentiment analysis pipeline (loaded once and reused)
# Downloads and prepares a neural network that has been trained to tell if a sentence is "Positive" or "Negative."
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# Improved 3-class sentiment model (POSITIVE / NEUTRAL / NEGATIVE).
# Built on RoBERTa and trained on tweets, so it handles informal text well
# and can express NEUTRAL — the default DistilBERT model cannot.
sentiment_analyzer_3class = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)

#The analyze_text(text) function takes a string of text and runs it through four steps: Tokenization, POS Tagging (Parts of Speech), Named Entity Recognition (NER), and Sentiment Analysis.
def analyze_text(text):
    """Run text through tokenization, POS tagging, NER, and sentiment analysis."""
    print("=" * 70)
    print(f"INPUT TEXT: {text}")
    print("=" * 70)

    # Step 2: Tokenization
    tokens = word_tokenize(text)
    print("\n[Step 2] Tokenization:")
    print(tokens)

    # Step 3: POS tagging
    # Identifies the role of each word. Is it a noun (NN), a verb (VB), or an adjective (JJ)
    tagged_tokens = nltk.pos_tag(tokens)
    print("\n[Step 3] POS Tagging:")
    print(tagged_tokens)

    # Step 4: Named Entity Recognition
    # This scans the text for "Entities." If it sees "Apple," it labels it ORG (Organization). If it sees "California," it labels it GPE (Geopolitical Entity/Location).
    doc = nlp(text)
    print("\n[Step 4] Named Entities:")
    if doc.ents:
        for ent in doc.ents:
            print(f"  {ent.text:<25} -> {ent.label_}")
    else:
        print("  (no entities found)")

    # Step 5: Sentiment analysis
    # The model reads the whole sentence and gives a score. For example: {'label': 'POSITIVE', 'score': 0.99}.
    result = sentiment_analyzer(text)
    print("\n[Step 5] Sentiment Analysis:")
    print(f"  {result}")
    print()

    return {
        "tokens": tokens,
        "pos_tags": tagged_tokens,
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "sentiment": result,
    }


def analyze_text_improved(text):
    """Improved pipeline addressing the Step 7 reflection points.

    Improvements over analyze_text():
    1. spaCy is used for tokenization AND POS tagging, so tokens line up
       with NER spans (no more NLTK/spaCy mismatch).
    2. Sentiment analysis uses a 3-class model (POS/NEU/NEG) so neutral
       text is not forced into a positive bucket.
    3. Long documents are split into sentences before sentiment scoring
       to stay under the 512-token transformer limit and to surface
       per-sentence tone shifts.
    """
    print("=" * 70)
    print(f"INPUT TEXT: {text}")
    print("=" * 70)

    # Single pass through spaCy — reused for tokens, POS, NER, and sentences.
    doc = nlp(text)

    # Step 2 (improved): tokenization via spaCy
    tokens = [token.text for token in doc]
    print("\n[Step 2] Tokenization (spaCy):")
    print(tokens)

    # Step 3 (improved): POS tagging via spaCy — same tokens as NER
    tagged_tokens = [(token.text, token.tag_) for token in doc]
    print("\n[Step 3] POS Tagging (spaCy):")
    print(tagged_tokens)

    # Step 4: Named Entity Recognition (unchanged, but now sharing tokens)
    print("\n[Step 4] Named Entities:")
    if doc.ents:
        for ent in doc.ents:
            print(f"  {ent.text:<25} -> {ent.label_}")
    else:
        print("  (no entities found)")

    # Step 5 (improved): per-sentence 3-class sentiment
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    print("\n[Step 5] Sentiment Analysis (3-class, per sentence):")
    sentence_results = []
    for sent in sentences:
        result = sentiment_analyzer_3class(sent)[0]
        sentence_results.append({"sentence": sent, **result})
        print(f"  [{result['label']:<8} {result['score']:.3f}] {sent}")

    return {
        "tokens": tokens,
        "pos_tags": tagged_tokens,
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "sentence_sentiment": sentence_results,
    }


# Step 6: Test the integrated system with multiple samples
if __name__ == "__main__":
    samples = [
        "Natural Language Processing is transforming AI applications.",
        "Serena Williams won Wimbledon in 2016, solidifying her status as one of the greatest tennis players in history.",
        "The new iPhone released by Apple in California was disappointing and overpriced.",
        "Elon Musk announced that SpaceX will launch a new rocket from Florida next month.",
    ]

    results = []
    for sample in samples:
        results.append(analyze_text(sample))

    # ----------------------------------------------------------------------
    # Step 7: Reflect and iterate
    # ----------------------------------------------------------------------
    # Reflection on the basic pipeline results:
    #
    # 1. Did the system correctly identify entities and classify sentiment?
    #    - Mostly yes, but spaCy's small model tagged "SpaceX" as PERSON
    #      (should be ORG) and "Natural Language Processing" as ORG
    #      (should not be an entity at all).
    #    - The default sentiment model is BINARY (positive/negative), so
    #      neutral/factual sentences are forced into "positive".
    #
    # 2. How could the integration be improved for more complex text inputs?
    #    - Use spaCy for BOTH tokenization and POS tagging so results stay
    #      consistent with NER (same tokenizer = aligned offsets).
    #    - Use a 3-class sentiment model that supports neutral.
    #    - Split long documents into sentences before sentiment scoring,
    #      because transformer models have a 512-token limit.
    #
    # 3. Were there any discrepancies between POS tagging and NER outputs?
    #    - Yes. NLTK tokenized "Natural Language Processing" as three
    #      separate NNP tokens, while spaCy NER merged them into one ORG
    #      entity. Using one tokenizer for everything removes this gap.
    #
    # The improved pipeline below implements these fixes.
    print("\n\n" + "#" * 70)
    print("# STEP 7: IMPROVED PIPELINE")
    print("#" * 70)
    for sample in samples:
        analyze_text_improved(sample)
