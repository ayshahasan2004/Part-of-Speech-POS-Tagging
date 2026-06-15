# ENCS5342 – Final Course Project
## Part-of-Speech (POS) Tagging | Track 1

**Birzeit University – Faculty of Engineering and Technology**  
**Department of Electrical and Computer Engineering**  
**Course:** ENCS5342 – Information Retrieval with Applications of NLP (Term 1252)

---

## Team Members

| Name | Student ID |
|------|------------|
| Aysha Hasan | 1220352 |
| Besan Maaly | 1222776 |

---

## Project Description

This project implements and evaluates Part-of-Speech (POS) taggers for English using the Penn Treebank corpus. POS tagging is the task of assigning a grammatical category (e.g., NOUN, VERB, ADJ) to each word in a sentence based on both its definition and its context.

The target label is a **joint coarse+fine-grained tag** (e.g., `NOUN-NNP`, `VERB-VBD`, `PRON-PRP`), combining the Universal POS tag (column 4) and the Penn Treebank POS tag (column 5) from the CoNLL-format dataset.

We implement and compare **at least two distinct approaches**, ranging from a simple baseline to a neural model, and report full evaluation metrics on both training and development sets.

---

## Repository Structure

```
.
├── README.md
├── notebooks/
│   └── ENCS5342_Project_Track1_1220352_1222776.ipynb   # Main Jupyter Notebook
├── src/
│   └── ...                                              # Helper scripts / utilities
├── data/
│   └── README.txt                                       # Instructions to download dataset
└── requirements.txt                                     # Python dependencies
```

> **Note:** Raw dataset files are not included in this repository. See `data/README.txt` for download instructions.

---

## Dataset

- **Corpus:** English Penn Treebank (Stanford-style dependency format, CoNLL columns)
- **Training set:** `en-universal-train.conll` (~37,840 sentences)
- **Validation set:** `en-universal-dev.conll` (~1,992 sentences)
- **Test set:** Held out by the instructor — not available to students

Dataset access: provided by the course instructor via Google Drive.

---

## Models & Approaches

*(To be updated as work progresses)*

- [ ] Baseline model (e.g., rule-based / logistic regression)
- [ ] Neural model (e.g., BiLSTM / Transformer encoder)

---

## Evaluation Metrics

For every model and every data split we report:

- Per-class Precision, Recall, and F1-score
- Micro-averaged Precision, Recall, and F1-score
- Macro-averaged Precision, Recall, and F1-score
- Overall token-level accuracy

---

## Setup & Installation

### Requirements

- Python 3.9+
- PyTorch
- See `requirements.txt` for full dependency list with versions

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/ENCS5342_Project_Track1_1220352_1222776.ipynb
```

Make sure all outputs are cleared before re-running from scratch to ensure full reproducibility. Random seeds are set explicitly throughout the notebook.

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Team & track registration | 9 June 2026 |
| Final notebook submission | 25 June 2026 (11:59 PM) |
| In-person discussion meetings | 27 June – 12 July 2026 |

---

## Submission

Final submission ZIP: `ENCS5342_Project_1_1220352_1222776.zip`  
Submitted via Moodle/ITC by **25 June 2026, 11:59 PM**.

---

