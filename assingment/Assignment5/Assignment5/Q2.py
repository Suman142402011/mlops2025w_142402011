# ---------------- Imports ---------------- #
import re
import wandb
from collections import Counter
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis

# ---------------- Initialize W&B ---------------- #
wandb.init(project="Q1-weak-supervision-ner", name="conll2003-lf-analysis-int")

# ---------------- Helper Function ---------------- #
def load_conll_file(file_path):
    """Load CoNLL-formatted file into list of sentences and NER tags"""
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence = []
        sentence_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
                continue
            parts = line.split()
            token = parts[0]
            ner_tag = parts[-1]
            sentence.append(token)
            sentence_labels.append(ner_tag)
        if sentence:
            sentences.append(sentence)
            labels.append(sentence_labels)
    return sentences, labels

# ---------------- Load Local Train Dataset ---------------- #
train_file = "D:\\Assignment5_mlops\\eng.train"
train_sentences, train_labels = load_conll_file(train_file)

# ---------------- Prepare DataFrame for LFs ---------------- #
train_texts = [" ".join(tokens) for tokens in train_sentences]

# Map string labels to integers for Snorkel
label_map = {"B-MISC": 0, "B-ORG": 1, "O": -1}  # ABSTAIN = -1
train_labels_int = [label_map.get(Counter(labels).most_common(1)[0][0], -1) for labels in train_labels]

df_train = pd.DataFrame({"text": train_texts, "label": train_labels_int})

# ---------------- Define Labeling Functions ---------------- #
ABSTAIN = -1
MISC = 0  # B-MISC mapped to 0
ORG = 1   # B-ORG mapped to 1

@labeling_function()
def lf_year(x):
    text = x.text
    if re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", text):
        return MISC
    return ABSTAIN

@labeling_function()
def lf_org_suffix(x):
    tokens = x.text.split()  # split sentence into tokens
    for token in tokens:
        if re.match(r".*(Inc\.|Corp\.|Ltd\.)$", token):
            return ORG
    return ABSTAIN


# ---------------- Apply Labeling Functions ---------------- #
lfs = [lf_year, lf_org_suffix]
applier = PandasLFApplier(lfs=lfs)
L = applier.apply(df_train)  # apply LFs to the DataFrame

# ---------------- Analyze LFs ---------------- #
analysis = LFAnalysis(L=L, lfs=lfs)
summary_df = analysis.lf_summary()  # Coverage, Overlaps, Conflicts

# Log LF coverage to W&B (skip accuracy)
for lf_name in summary_df.index:
    wandb.log({
        f"{lf_name}_coverage": summary_df.loc[lf_name, "Coverage"]
    })

# Print for debug
print("Labeling Function Summary:")
print(summary_df)


wandb.finish()
