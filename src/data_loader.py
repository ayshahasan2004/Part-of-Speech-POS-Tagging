"""
data_loader.py

Reads the Penn Treebank dataset in CoNLL column format (as specified in the
ENCS5342 project description) and converts it into a list of (words, tags)
pairs ready for use by the HMM baseline and the BiLSTM model.

CoNLL column reference (per the project spec):
    1  Word ID                - integer index, starts at 1 per sentence
    2  Word Form               - surface form of the token
    3  Lemma                   - NOT USED in this project
    4  Coarse-grained POS      - Universal POS tag (NOUN, VERB, DET, ADP, ...)
    5  Fine-grained POS        - Penn Treebank POS tag (NN, VBD, DT, IN, ...)
    6  Features                - NOT USED in this project
    7  Head ID                 - NOT USED in this project
    8  Dependency Relation     - NOT USED in this project
    9-10 Projective fields     - NOT USED in this project

Target label: the project requires predicting a JOINT tag that merges the
coarse- and fine-grained POS tags, e.g. NOUN-NNP, VERB-VBD, PRON-PRP, ADP-IN.
This loader builds that joint tag as "{coarse_pos}-{fine_pos}".

Sentences in the file are separated by a single blank line.
"""

from pathlib import Path


def load_conll(filepath):
    """
    Parse a CoNLL-format file into a list of (words, tags) tuples.

    Each sentence becomes one tuple:
        words: list[str]  - the surface form of every token in the sentence
        tags:  list[str]  - the joint POS tag "{CPOS}-{FPOS}" for every token

    Args:
        filepath (str or Path): path to a .conll file
            (e.g. "data/en-universal-train.conll")

    Returns:
        list[tuple[list[str], list[str]]]: one (words, tags) pair per sentence
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find data file: {filepath}")

    sentences = []
    words = []
    tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Blank line marks the end of a sentence
            if line == "":
                if words:
                    sentences.append((words, tags))
                    words = []
                    tags = []
                continue

            cols = line.split("\t")

            # Defensive check: a valid CoNLL line must have at least the
            # 5 columns we need (ID, Form, Lemma, CPOS, FPOS)
            if len(cols) < 5:
                raise ValueError(
                    f"Malformed line {line_num} in {filepath.name}: "
                    f"expected at least 5 tab-separated columns, got {len(cols)} "
                    f"-> {raw_line!r}"
                )

            word_form = cols[1]
            coarse_pos = cols[3]
            fine_pos = cols[4]
            joint_tag = f"{coarse_pos}-{fine_pos}"

            words.append(word_form)
            tags.append(joint_tag)

        # Handle the case where the file does not end with a blank line
        if words:
            sentences.append((words, tags))

    return sentences


def get_vocab_and_tag_sets(sentences):
    """
    Collect the set of unique words and unique joint tags appearing in a
    list of (words, tags) sentence tuples. Useful for building word2idx /
    tag2idx mappings and for sanity-checking the tag set against the
    project's example tags (NOUN-NNP, VERB-VBD, PRON-PRP, ADP-IN, etc.).

    Args:
        sentences: output of load_conll()

    Returns:
        (set[str], set[str]): (unique_words, unique_tags)
    """
    unique_words = set()
    unique_tags = set()
    for words, tags in sentences:
        unique_words.update(words)
        unique_tags.update(tags)
    return unique_words, unique_tags


if __name__ == "__main__":
    train_path = "../data/en-universal-train.conll"
    dev_path = "../data/en-universal-dev.conll"

    train_sentences = load_conll(train_path)
    dev_sentences = load_conll(dev_path)

    print(f"Number of training sentences: {len(train_sentences)}")
    print(f"Number of dev sentences:      {len(dev_sentences)}\n")

    train_vocab, train_tags = get_vocab_and_tag_sets(train_sentences)
    print(f"Training vocabulary size: {len(train_vocab)}")
    print(f"Number of distinct joint tags: {len(train_tags)}\n")

    print("First training sentence:")
    words, tags = train_sentences[0]
    for w, t in zip(words, tags):
        print(f"  {w:15s} -> {t}")