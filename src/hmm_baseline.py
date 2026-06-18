"""
hmm_baseline.py

Implements an HMM POS tagger trained from scratch on the CoNLL-format
Penn Treebank data. Computes:
    - Initial probabilities  (pi):  P(tag is the first tag of a sentence)
    - Transition probabilities (A): P(tag_i | tag_{i-1})
    - Emission probabilities   (B): P(word | tag)

Smoothing:
    - Initial probabilities use Laplace (add-1) smoothing over the small
      tag set (46 tags), so add-1 is cheap and safe here.
    - Transition and emission probabilities use add-k smoothing with a
      small k (default 0.1), because add-1 over a 43k-word vocabulary
      would assign far too much probability mass to combinations that
      never occurred.

Unknown word handling:
    - Any training word that occurs only once (a "singleton") is treated
      as <UNK> when building emission counts. This lets the model learn
      a realistic emission distribution for <UNK>, which is reused for
      any genuinely unseen word encountered at dev/test time.

Log-space probabilities:
    - Multiplying many raw probabilities together (one per token in a
      sentence) causes numerical underflow: the running product becomes
      so small that floating point rounds it to exactly 0.0, which makes
      it impossible to compare different tag sequences. This is fatal for
      Viterbi, which must compare thousands of candidate sequences.
    - The fix is to work in log-space: log(a * b * c) = log(a) + log(b) +
      log(c). Summing logs never underflows the way multiplying raw
      probabilities does, so log_initial_prob / log_transition_prob /
      log_emission_prob are the versions Viterbi will actually use.
"""

import math
from collections import defaultdict, Counter

from data_loader import load_conll

UNK_TOKEN = "<UNK>"


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
        """
        Train the HMM by counting from the data.

        Args:
            sentences: list of (words, tags) tuples, as returned by
                       data_loader.load_conll()
        """
        # Step 1: find singleton words (occur exactly `rare_threshold` times
        # or fewer) across the WHOLE training set, before counting anything
        word_freq = Counter()
        for words, _ in sentences:
            word_freq.update(words)
        self.rare_words = {w for w, c in word_freq.items() if c <= self.rare_threshold}

        # Step 2: accumulate counts, replacing rare words with <UNK>
        for words, tags in sentences:
            self.sentence_count += 1
            prev_tag = None

            for i, (word, tag) in enumerate(zip(words, tags)):
                observed_word = UNK_TOKEN if word in self.rare_words else word

                self.tags.add(tag)
                self.vocab.add(observed_word)
                self.tag_counts[tag] += 1
                self.emission_counts[tag][observed_word] += 1

                if i == 0:
                    self.initial_counts[tag] += 1
                else:
                    self.transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        self.vocab.add(UNK_TOKEN)
        self.num_tags = len(self.tags)
        self.vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # Raw probabilities (useful for teaching / explaining the math,
    # but NOT safe to use directly inside Viterbi -- see log_* versions)
    # ------------------------------------------------------------------

    def initial_prob(self, tag):
        """pi(tag): Laplace-smoothed probability that `tag` starts a sentence."""
        return (self.initial_counts[tag] + 1) / (self.sentence_count + self.num_tags)

    def transition_prob(self, prev_tag, tag):
        """A(tag | prev_tag): add-k smoothed transition probability."""
        prev_total = sum(self.transition_counts[prev_tag].values())
        return (self.transition_counts[prev_tag][tag] + self.k_transition) / \
               (prev_total + self.k_transition * self.num_tags)

    def emission_prob(self, tag, word):
        """B(word | tag): add-k smoothed emission probability.
        Unseen words are mapped to <UNK> automatically."""
        observed_word = word if word in self.vocab else UNK_TOKEN
        tag_total = self.tag_counts[tag]
        return (self.emission_counts[tag][observed_word] + self.k_emission) / \
               (tag_total + self.k_emission * self.vocab_size)

    # ------------------------------------------------------------------
    # Log-space probabilities -- USE THESE for Viterbi and for scoring
    # full sentences. They avoid numerical underflow when many small
    # probabilities are combined across a long sequence of tokens.
    # ------------------------------------------------------------------

    def log_initial_prob(self, tag):
        """log(pi(tag)). Numerically safe version of initial_prob()."""
        return math.log(self.initial_prob(tag))

    def log_transition_prob(self, prev_tag, tag):
        """log(A(tag | prev_tag)). Numerically safe version of transition_prob()."""
        return math.log(self.transition_prob(prev_tag, tag))

    def log_emission_prob(self, tag, word):
        """log(B(word | tag)). Numerically safe version of emission_prob()."""
        return math.log(self.emission_prob(tag, word))


def explain_sentence(hmm, words, tags):
    """
    Walks through the HMM probability calculation step-by-step for ONE
    specific sentence, using its known (gold) tags. This only SCORES the
    given sequence -- it does not search for the best one. Viterbi (next
    step) is what performs that search.

    NOTE: this multiplies raw probabilities together, which is fine for a
    short demonstration but will underflow to 0.0 on longer sentences.
    See explain_sentence_log() below for the numerically safe version.
    """
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
    """
    Same as explain_sentence(), but accumulates log-probabilities by
    summing instead of multiplying raw probabilities. This is the
    numerically safe version that Viterbi will actually use, since
    log(a * b * c) = log(a) + log(b) + log(c) and summing logs does not
    underflow the way multiplying many small raw probabilities does.
    """
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

    print(f"Number of tags: {hmm.num_tags}")
    print(f"Vocabulary size (incl. <UNK>): {hmm.vocab_size}")
    print(f"Number of singleton (rare) words mapped to <UNK>: {len(hmm.rare_words)}\n")

    print("Sample initial probabilities:")
    for tag in ["DET-DT", "NOUN-NNP", "VERB-VBD", "ADP-IN"]:
        print(f"  pi({tag}) = {hmm.initial_prob(tag):.6f}")

    print("\nSample transition probabilities:")
    print(f"  A(NOUN-NN | DET-DT)  = {hmm.transition_prob('DET-DT', 'NOUN-NN'):.6f}")
    print(f"  A(VERB-VBD | NOUN-NNP) = {hmm.transition_prob('NOUN-NNP', 'VERB-VBD'):.6f}")

    print("\nSample emission probabilities:")
    print(f"  B('the'  | DET-DT)   = {hmm.emission_prob('DET-DT', 'the'):.6f}")
    print(f"  B('dog'  | NOUN-NN)  = {hmm.emission_prob('NOUN-NN', 'dog'):.6f}")
    print(f"  B('xyzabc' | NOUN-NN) = {hmm.emission_prob('NOUN-NN', 'xyzabc'):.6f}  (unseen word -> <UNK>)")

    print("\n" + "=" * 60)
    print("APPLYING THE HMM TO A REAL SENTENCE (raw probabilities)")
    print("=" * 60 + "\n")
    words, tags = train_sentences[0]
    explain_sentence(hmm, words, tags)

    print("\n" + "=" * 60)
    print("SAME SENTENCE, USING LOG-SPACE (avoids numerical underflow)")
    print("=" * 60 + "\n")
    explain_sentence_log(hmm, words, tags)