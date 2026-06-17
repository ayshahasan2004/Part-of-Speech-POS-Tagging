"""
vocab.py

Builds word2idx and tag2idx mappings from the training data vocabulary.
These mappings are SHARED between the HMM baseline and the BiLSTM model,
so both team members must use the exact same vocab.py output to ensure
fair, consistent evaluation.

<PAD> (index 0) - padding token, used by the BiLSTM for batching sentences
                  of different lengths. Not actively used by the HMM, but
                  kept in the mapping for consistency between both models.
<UNK> (index 1) - unknown word token, used for any word seen at dev/test
                  time that did not appear in the training vocabulary.
"""

from data_loader import load_conll, get_vocab_and_tag_sets


def build_word2idx(vocab):
    """
    Build a word-to-index mapping from a vocabulary set.

    Args:
        vocab (set[str]): unique words from the training set

    Returns:
        dict[str, int]: word2idx mapping, starting with <PAD>=0, <UNK>=1
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(vocab):
        word2idx[word] = len(word2idx)
    return word2idx


def build_tag2idx(tags):
    """
    Build a tag-to-index mapping from a set of joint POS tags.

    Args:
        tags (set[str]): unique joint tags from the training set

    Returns:
        dict[str, int]: tag2idx mapping, starting with <PAD>=0
    """
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