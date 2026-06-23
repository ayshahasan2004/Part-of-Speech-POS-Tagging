import math
from collections import defaultdict, Counter

from data_loader import load_conll

UNK_TOKEN = "<UNK>"


def get_unk_class(word):
    """Map unseen words to feature-based unknown classes."""
    if word[0].isupper():                return "<UNK-CAP>"
    elif word.endswith("ing"):           return "<UNK-ING>"
    elif word.endswith("ed"):            return "<UNK-ED>"
    elif word.endswith("ly"):            return "<UNK-LY>"
    elif word.endswith("er"):            return "<UNK-ER>"
    elif word.endswith("est"):           return "<UNK-EST>"
    elif word.endswith("ion"):           return "<UNK-ION>"
    elif word.endswith("ness"):          return "<UNK-NESS>"
    elif any(c.isdigit() for c in word): return "<UNK-NUM>"
    elif "-" in word:                    return "<UNK-HYPH>"
    else:                                return UNK_TOKEN


class HMMTagger:
    def __init__(self, k_emission=0.1, k_transition=0.1, rare_threshold=1):
        self.k_emission = k_emission
        self.k_transition = k_transition
        self.rare_threshold = rare_threshold

        self.tags = set()
        self.vocab = set()
        self.rare_words = set()

        self.initial_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.sentence_count = 0

        self.num_tags = 0
        self.vocab_size = 0

    def fit(self, sentences):
        """Train the HMM by collecting initial, transition, and emission counts."""
        word_freq = Counter()
        for words, _ in sentences:
            word_freq.update(words)
        self.rare_words = {w for w, c in word_freq.items() if c <= self.rare_threshold}

        # Rare words are replaced with UNK classes so unseen words get useful probabilities.
        for words, tags in sentences:
            self.sentence_count += 1
            prev_tag = None

            for i, (word, tag) in enumerate(zip(words, tags)):
                observed_word = get_unk_class(word) if word in self.rare_words else word

                self.tags.add(tag)
                self.vocab.add(observed_word)
                self.tag_counts[tag] += 1
                self.emission_counts[tag][observed_word] += 1

                if i == 0:
                    self.initial_counts[tag] += 1
                else:
                    self.transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        # Plain <UNK> is the fallback when no surface feature matches.
        self.vocab.add(UNK_TOKEN)
        self.num_tags = len(self.tags)
        self.vocab_size = len(self.vocab)

    def initial_prob(self, tag):
        return (self.initial_counts[tag] + 1) / (self.sentence_count + self.num_tags)

    def transition_prob(self, prev_tag, tag):
        prev_total = sum(self.transition_counts[prev_tag].values())
        return (self.transition_counts[prev_tag][tag] + self.k_transition) / \
               (prev_total + self.k_transition * self.num_tags)

    def emission_prob(self, tag, word):
        """Return the smoothed probability of a word given a tag."""
        if word in self.vocab:
            observed_word = word
        elif word.lower() in self.vocab:
            observed_word = word.lower()
        else:
            observed_word = get_unk_class(word)

        tag_total = self.tag_counts[tag]
        return (self.emission_counts[tag][observed_word] + self.k_emission) / \
               (tag_total + self.k_emission * self.vocab_size)

    def log_initial_prob(self, tag):
        return math.log(self.initial_prob(tag))

    def log_transition_prob(self, prev_tag, tag):
        return math.log(self.transition_prob(prev_tag, tag))

    def log_emission_prob(self, tag, word):
        return math.log(self.emission_prob(tag, word))


def explain_sentence(hmm, words, tags):
    """Print the raw-probability score for one known tag sequence."""
    print(f"Sentence: {' '.join(words)}\n")

    score = hmm.initial_prob(tags[0])
    print(f"pi({tags[0]}) = {score:.6f}")

    for i in range(len(words)):
        b = hmm.emission_prob(tags[i], words[i])
        score *= b
        print(f"  B('{words[i]}' | {tags[i]}) = {b:.6f}   -> running product = {score:.3e}")

        if i + 1 < len(words):
            a = hmm.transition_prob(tags[i], tags[i + 1])
            score *= a
            print(f"  A({tags[i+1]} | {tags[i]}) = {a:.6f}   -> running product = {score:.3e}")

    print(f"\nFinal score for this exact gold tag sequence: {score:.3e}")
    return score


def explain_sentence_log(hmm, words, tags):
    """Print the log-probability score for one known tag sequence."""
    print(f"Sentence: {' '.join(words)}\n")

    log_score = hmm.log_initial_prob(tags[0])
    print(f"log pi({tags[0]}) = {log_score:.4f}")

    for i in range(len(words)):
        log_b = hmm.log_emission_prob(tags[i], words[i])
        log_score += log_b
        print(f"  log B('{words[i]}' | {tags[i]}) = {log_b:.4f}   -> running sum = {log_score:.4f}")

        if i + 1 < len(words):
            log_a = hmm.log_transition_prob(tags[i], tags[i + 1])
            log_score += log_a
            print(f"  log A({tags[i+1]} | {tags[i]}) = {log_a:.4f}   -> running sum = {log_score:.4f}")

    print(f"\nFinal LOG score for this exact gold tag sequence: {log_score:.4f}")
    print(f"(equivalent raw probability would be: {math.exp(log_score):.3e})")
    return log_score


if __name__ == "__main__":
    train_sentences = load_conll("../data/en-universal-train.conll")

    hmm = HMMTagger()
    hmm.fit(train_sentences)

    print(f"Number of tags:                          {hmm.num_tags}")
    print(f"Vocabulary size (incl. UNK classes):     {hmm.vocab_size}")
    print(f"Number of rare words mapped to UNK class:{len(hmm.rare_words)}\n")

    print("Sample initial probabilities:")
    for tag in ["DET-DT", "NOUN-NNP", "VERB-VBD", "ADP-IN"]:
        print(f"  pi({tag}) = {hmm.initial_prob(tag):.6f}")

    print("\nSample transition probabilities:")
    print(f"  A(NOUN-NN | DET-DT)    = {hmm.transition_prob('DET-DT', 'NOUN-NN'):.6f}")
    print(f"  A(VERB-VBD | NOUN-NNP) = {hmm.transition_prob('NOUN-NNP', 'VERB-VBD'):.6f}")

    print("\nSample emission probabilities:")
    print(f"  B('the'          | DET-DT)  = {hmm.emission_prob('DET-DT', 'the'):.6f}")
    print(f"  B('dog'          | NOUN-NN) = {hmm.emission_prob('NOUN-NN', 'dog'):.6f}")
    print(f"  B('Was'          | VERB-VBD)= {hmm.emission_prob('VERB-VBD', 'Was'):.6f}  (caps -> lowercase)")
    print(f"  B('policy-making'| VERB-VBG)= {hmm.emission_prob('VERB-VBG', 'policy-making'):.6f}  (unknown -> <UNK-HYPH>)")
    print(f"  B('quickly'      | ADV-RB)  = {hmm.emission_prob('ADV-RB', 'quicklyxyz'):.6f}  (unknown -> <UNK-LY>)")

    print("\n" + "=" * 60)
    print("APPLYING THE HMM TO A REAL SENTENCE (raw probabilities)")
    print("=" * 60 + "\n")
    words, tags = train_sentences[0]
    explain_sentence(hmm, words, tags)

    print("\n" + "=" * 60)
    print("SAME SENTENCE, USING LOG-SPACE (avoids numerical underflow)")
    print("=" * 60 + "\n")
    explain_sentence_log(hmm, words, tags)
