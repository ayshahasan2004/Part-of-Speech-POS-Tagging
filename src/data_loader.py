from pathlib import Path


def load_conll(filepath):
    """Parse a CoNLL file into (words, joint_tags) sentence tuples."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find data file: {filepath}")

    sentences = []
    words = []
    tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # A blank line separates sentences in CoNLL files.
            if line == "":
                if words:
                    sentences.append((words, tags))
                    words = []
                    tags = []
                continue

            cols = line.split("\t")

            # The project needs word form, coarse POS, and fine POS columns.
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

        # Keep the last sentence when the file has no trailing blank line.
        if words:
            sentences.append((words, tags))

    return sentences


def get_vocab_and_tag_sets(sentences):
    """Collect unique words and joint POS tags from loaded sentences."""
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
