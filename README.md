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

## GitHub Repository
https://github.com/ayshahasan2004/Part-of-Speech-POS-Tagging

---

## Project Description

This project implements and evaluates Part-of-Speech (POS) taggers for English using the Penn Treebank corpus. POS tagging is the task of assigning a grammatical category (e.g., NOUN, VERB, ADJ) to each word in a sentence based on both its definition and its context.

The target label is a **joint coarse+fine-grained tag** (e.g., `NOUN-NNP`, `VERB-VBD`, `PRON-PRP`), combining the Universal POS tag (column 4) and the Penn Treebank POS tag (column 5) from the CoNLL-format dataset.

We implement and compare **two distinct approaches** and report full evaluation metrics on both training and development sets.

---

## Repository Structure
├── README.md

├── notebooks/

│   └── ENCS5342_Project_Track1_1220352_1222776.ipynb

├── src/

│   ├── data_loader.py

│   ├── vocab.py

│   ├── hmm_baseline.py

│   ├── viterbi.py

│   ├── evaluation.py

│   ├── bilstm_model.py

│   ├── train_bilstm.py

│   └── train_utils.py

├── data/

│   ├── en-universal-train.conll

│   └── en-universal-dev.conll

└── requirements.txt

---

## Dataset

- **Corpus:** English Penn Treebank (Stanford-style dependency format, CoNLL columns)
- **Training set:** `en-universal-train.conll` (37,840 sentences)
- **Validation set:** `en-universal-dev.conll` (1,992 sentences)
- **Test set:** Held out by the instructor — not available to students

Dataset access: provided by the course instructor via Google Drive.

---

## Models & Approaches

- [x] **HMM Baseline** with Viterbi decoding — Dev Accuracy: **95.51%**
- [x] **BiLSTM Neural Tagger** — Dev Accuracy: **95.12%**

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
- PyTorch 2.0+

### Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Run the notebook

```bash
jupyter notebook notebooks/ENCS5342_Project_Track1_1220352_1222776.ipynb
```

Make sure to run all cells in order. Random seeds are set explicitly throughout the notebook for reproducibility.

---

## Key Results

| Model | Train Accuracy | Dev Accuracy | Dev Micro F1 | Dev Macro F1 |
|-------|---------------|--------------|--------------|--------------|
| HMM Baseline | 95.99% | 95.51% | 0.9551 | 0.8769 |
| BiLSTM | 98.50% | 95.12% | 0.9512 | 0.8682 |

---