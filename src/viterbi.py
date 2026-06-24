import math

import numpy as np

from hmm_baseline import HMMTagger
from data_loader import load_conll


def _prepare_viterbi_cache(hmm):
    """Precompute tag lists and transition scores once instead of inside every sentence."""
    if hasattr(hmm, "_viterbi_cache"):
        return hmm._viterbi_cache

    tags = sorted(hmm.tags)
    n_tags = len(tags)

    initial_log_probs = np.array(
        [hmm.log_initial_prob(tag) for tag in tags],
        dtype=np.float64
    )

    transition_log_probs = np.empty((n_tags, n_tags), dtype=np.float64)
    for prev_i, prev_tag in enumerate(tags):
        for tag_i, tag in enumerate(tags):
            transition_log_probs[prev_i, tag_i] = hmm.log_transition_prob(prev_tag, tag)

    hmm._viterbi_cache = {
        "tags": tags,
        "initial_log_probs": initial_log_probs,
        "transition_log_probs": transition_log_probs,
    }
    return hmm._viterbi_cache


def viterbi(hmm, words):
    """Return the most likely tag sequence for one sentence using vectorized Viterbi."""
    cache = _prepare_viterbi_cache(hmm)
    tags = cache["tags"]
    initial_log_probs = cache["initial_log_probs"]
    transition_log_probs = cache["transition_log_probs"]

    n_words = len(words)
    n_tags = len(tags)

    dp = np.full((n_words, n_tags), -np.inf, dtype=np.float64)
    bp = np.zeros((n_words, n_tags), dtype=np.int32)

    first_emissions = np.array(
        [hmm.log_emission_prob(tag, words[0]) for tag in tags],
        dtype=np.float64
    )
    dp[0] = initial_log_probs + first_emissions
    bp[0] = -1

    for t in range(1, n_words):
        log_emit = np.array(
            [hmm.log_emission_prob(tag, words[t]) for tag in tags],
            dtype=np.float64
        )

        scores = dp[t - 1][:, None] + transition_log_probs
        bp[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_emit

    best_last_idx = int(np.argmax(dp[n_words - 1]))
    best_score = float(dp[n_words - 1, best_last_idx])

    best_tag_indices = [0] * n_words
    best_tag_indices[n_words - 1] = best_last_idx

    for t in range(n_words - 2, -1, -1):
        best_tag_indices[t] = int(bp[t + 1, best_tag_indices[t + 1]])

    best_tags = [tags[i] for i in best_tag_indices]
    return best_tags, best_score


def predict(hmm, sentences, progress_every=1000):
    """Run Viterbi on every sentence and return predicted tag sequences."""
    predictions = []
    total = len(sentences)

    for i, (words, _) in enumerate(sentences, start=1):
        predictions.append(viterbi(hmm, words)[0])
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  decoded {i}/{total} sentences")

    return predictions


if __name__ == "__main__":
    print("Loading data...")
    train_sentences = load_conll("../data/en-universal-train.conll")
    dev_sentences   = load_conll("../data/en-universal-dev.conll")

    print("Training HMM...")
    hmm = HMMTagger()
    hmm.fit(train_sentences)

    print("\n" + "=" * 60)
    print("VITERBI DEMO - first 3 dev sentences")
    print("=" * 60)

    for i, (words, gold_tags) in enumerate(dev_sentences[:3]):
        pred_tags, score = viterbi(hmm, words)

        print(f"\nSentence {i + 1}: {' '.join(words)}")
        print(f"Log-score of predicted sequence: {score:.4f}\n")
        print(f"  {'Word':<15} {'Gold':<15} {'Predicted':<15} {'Match'}")
        print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*5}")
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            match = "yes" if gold == pred else "no"
            print(f"  {word:<15} {gold:<15} {pred:<15} {match}")

    print("\n" + "=" * 60)
    print("QUICK ACCURACY CHECK - first 100 dev sentences")
    print("=" * 60)
    correct = 0
    total   = 0
    for words, gold_tags in dev_sentences[:100]:
        pred_tags, _ = viterbi(hmm, words)
        for g, p in zip(gold_tags, pred_tags):
            correct += (g == p)
            total   += 1
    print(f"Accuracy on first 100 dev sentences: {correct}/{total} = {correct/total:.4f}")
