from data_loader import load_conll, get_vocab_and_tag_sets


def build_word2idx(vocab):
    """Build a word index with reserved padding and unknown tokens."""
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(vocab):
        word2idx[word] = len(word2idx)
    return word2idx


def build_tag2idx(tags):
    """Build a tag index with padding reserved at index 0."""
    tag2idx = {"<PAD>": 0}
    for tag in sorted(tags):
        tag2idx[tag] = len(tag2idx)
    return tag2idx


if __name__ == "__main__":
    train_sentences = load_conll("../data/en-universal-train.conll")
    vocab, tags = get_vocab_and_tag_sets(train_sentences)

    word2idx = build_word2idx(vocab)
    tag2idx = build_tag2idx(tags)

    print(f"Vocabulary size (including <PAD>, <UNK>): {len(word2idx)}")
    print(f"Number of tags (including <PAD>): {len(tag2idx)}\n")

    print("Sample word2idx entries:")
    for word in list(word2idx.keys())[:10]:
        print(f"  {word!r:15s} -> {word2idx[word]}")

    print("\nAll tag2idx entries:")
    for tag, idx in tag2idx.items():
        print(f"  {tag:12s} -> {idx}")
