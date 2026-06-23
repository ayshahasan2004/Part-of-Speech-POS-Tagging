import math
from hmm_baseline import HMMTagger
from data_loader import load_conll


def viterbi(hmm, words):
    """Return the most likely tag sequence for one sentence."""
    tags = sorted(hmm.tags)
    n_words = len(words)
    n_tags = len(tags)
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}

    # dp stores best log scores; bp stores backpointers for reconstruction.
    dp = [[float("-inf")] * n_tags for _ in range(n_words)]
    bp = [[0] * n_tags for _ in range(n_words)]

    for i, tag in enumerate(tags):
        dp[0][i] = hmm.log_initial_prob(tag) + hmm.log_emission_prob(tag, words[0])
        bp[0][i] = -1

    for t in range(1, n_words):
        for i, tag in enumerate(tags):
            log_emit = hmm.log_emission_prob(tag, words[t])

            best_score = float("-inf")
            best_prev  = 0

            for j, prev_tag in enumerate(tags):
                score = dp[t - 1][j] + hmm.log_transition_prob(prev_tag, tag) + log_emit
                if score > best_score:
                    best_score = score
                    best_prev  = j

            dp[t][i] = best_score
            bp[t][i] = best_prev

    best_last_idx = max(range(n_tags), key=lambda i: dp[n_words - 1][i])
    best_score    = dp[n_words - 1][best_last_idx]

    # Follow backpointers from the best final tag to recover the sequence.
    best_tag_indices = [0] * n_words
    best_tag_indices[n_words - 1] = best_last_idx

    for t in range(n_words - 2, -1, -1):
        best_tag_indices[t] = bp[t + 1][best_tag_indices[t + 1]]

    best_tags = [tags[i] for i in best_tag_indices]
    return best_tags, best_score


def predict(hmm, sentences):
    """Run Viterbi on every sentence and return predicted tag sequences."""
    return [viterbi(hmm, words)[0] for words, _ in sentences]


if __name__ == "__main__":
    print("Loading data...")
    train_sentences = load_conll("../data/en-universal-train.conll")
    dev_sentences   = load_conll("../data/en-universal-dev.conll")

    print("Training HMM...")
    hmm = HMMTagger()
    hmm.fit(train_sentences)

    print("\n" + "=" * 60)
    print("VITERBI DEMO — first 3 dev sentences")
    print("=" * 60)

    for i, (words, gold_tags) in enumerate(dev_sentences[:3]):
        pred_tags, score = viterbi(hmm, words)

        print(f"\nSentence {i + 1}: {' '.join(words)}")
        print(f"Log-score of predicted sequence: {score:.4f}\n")
        print(f"  {'Word':<15} {'Gold':<15} {'Predicted':<15} {'Match'}")
        print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*5}")
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            match = "✓" if gold == pred else "✗"
            print(f"  {word:<15} {gold:<15} {pred:<15} {match}")

    print("\n" + "=" * 60)
    print("QUICK ACCURACY CHECK — first 100 dev sentences")
    print("=" * 60)
    correct = 0
    total   = 0
    for words, gold_tags in dev_sentences[:100]:
        pred_tags, _ = viterbi(hmm, words)
        for g, p in zip(gold_tags, pred_tags):
            correct += (g == p)
            total   += 1
    print(f"Accuracy on first 100 dev sentences: {correct}/{total} = {correct/total:.4f}")
