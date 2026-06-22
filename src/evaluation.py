from collections import defaultdict


def evaluate(gold_sentences, predicted_tag_sequences):
    """Compute accuracy plus per-class, micro, and macro metrics."""
    # Count true positives, false positives, and false negatives per tag.
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    total_tokens  = 0
    correct_tokens = 0

    for (_, gold_tags), pred_tags in zip(gold_sentences, predicted_tag_sequences):
        for gold, pred in zip(gold_tags, pred_tags):
            total_tokens += 1
            support[gold] += 1

            if gold == pred:
                correct_tokens += 1
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

    all_tags = sorted(support.keys())

    per_class = {}
    for tag in all_tags:
        p = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0.0
        r = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0.0
        f = (2 * p * r) / (p + r)         if (p + r) > 0              else 0.0
        per_class[tag] = {
            'precision': p,
            'recall':    r,
            'f1':        f,
            'support':   support[tag]
        }

    # Micro averages combine counts before computing the metrics.
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro averages give each tag the same weight.
    macro_p = sum(per_class[t]['precision'] for t in all_tags) / len(all_tags)
    macro_r = sum(per_class[t]['recall']    for t in all_tags) / len(all_tags)
    macro_f = sum(per_class[t]['f1']        for t in all_tags) / len(all_tags)

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'accuracy':  accuracy,
        'per_class': per_class,
        'micro':     {'precision': micro_p, 'recall': micro_r, 'f1': micro_f},
        'macro':     {'precision': macro_p, 'recall': macro_r, 'f1': macro_f},
    }


def print_report(results, title="Evaluation Report"):
    """Print a formatted evaluation report to the console."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    print(f"\n  Overall Accuracy: {results['accuracy']:.4f} "
          f"({results['accuracy']*100:.2f}%)\n")

    print(f"  {'Tag':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for tag, m in sorted(results['per_class'].items()):
        print(f"  {tag:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")

    print(f"\n  {'-'*57}")

    mi = results['micro']
    ma = results['macro']
    print(f"  {'Micro avg':<15} {mi['precision']:>10.4f} {mi['recall']:>10.4f} {mi['f1']:>10.4f}")
    print(f"  {'Macro avg':<15} {ma['precision']:>10.4f} {ma['recall']:>10.4f} {ma['f1']:>10.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    from data_loader import load_conll
    from hmm_baseline import HMMTagger
    from viterbi import predict

    print("Loading data...")
    train_sentences = load_conll("../data/en-universal-train.conll")
    dev_sentences   = load_conll("../data/en-universal-dev.conll")

    print("Training HMM...")
    hmm = HMMTagger()
    hmm.fit(train_sentences)

    print("Running Viterbi on training set (this may take a few minutes)...")
    train_preds = predict(hmm, train_sentences)
    train_results = evaluate(train_sentences, train_preds)
    print_report(train_results, title="HMM Baseline — Training Set")

    print("Running Viterbi on dev set...")
    dev_preds   = predict(hmm, dev_sentences)
    dev_results = evaluate(dev_sentences, dev_preds)
    print_report(dev_results, title="HMM Baseline — Dev Set")
