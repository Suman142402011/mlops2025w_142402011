# ---------------- Imports ---------------- #
import re
import wandb
import pandas as pd
from collections import Counter
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

# ---------------- Initialize W&B ---------------- #
wandb.init(project="Q1-weak-supervision-ner", name="conll2003-lf-aggregation")

# ---------------- Helper Function ---------------- #
def load_conll_file(file_path):
    """Load CoNLL-formatted file into list of sentences and NER tags"""
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence, sentence_labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence, sentence_labels = [], []
                continue
            parts = line.split()
            token, ner_tag = parts[0], parts[-1]
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
train_labels_simple = [Counter(labels).most_common(1)[0][0] for labels in train_labels]
df_train = pd.DataFrame({"text": train_texts, "label": train_labels_simple})

# ---------------- Define Labeling Functions ---------------- #
ABSTAIN = -1
MISC = 0
ORG = 1

@labeling_function()
def lf_year(x):
    if re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", x.text):
        return MISC
    return ABSTAIN

@labeling_function()
def lf_org_suffix(x):
    tokens = x.text.split()
    for token in tokens:
        if re.match(r".*(Inc\.|Corp\.|Ltd\.)$", token):
            return ORG
    return ABSTAIN

# ---------------- Apply Labeling Functions ---------------- #
lfs = [lf_year, lf_org_suffix]
applier = PandasLFApplier(lfs=lfs)
L = applier.apply(df_train)

# ---------------- Analyze LFs ---------------- #
analysis = LFAnalysis(L=L, lfs=lfs)
summary_df = analysis.lf_summary()
print("Labeling Function Summary:")
print(summary_df)

# Log coverage to W&B
for lf_name in summary_df.index:
    wandb.log({
        f"{lf_name}_coverage": summary_df.loc[lf_name, "Coverage"]
    })

# ---------------- Majority Label Voter ---------------- #
voter = MajorityLabelVoter(cardinality=2)  # 2 classes: MISC=0, ORG=1
aggregated_labels = voter.predict(L=L)

# ---------------- Add aggregated labels to DataFrame ---------------- #
df_train["aggregated_label"] = aggregated_labels

# ---------------- Log Aggregated Label Stats ---------------- #
agg_counts = df_train["aggregated_label"].value_counts()
# ---------------- Log Aggregated Label Stats ---------------- #
agg_counts_dict = {str(k): int(v) for k, v in agg_counts.items()}  # keys as strings
wandb.summary["aggregated_label_distribution"] = agg_counts_dict


print("\nAggregated Label Distribution:")
print(agg_counts)

wandb.finish()
