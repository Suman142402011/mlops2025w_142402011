# ---------------- Imports ---------------- #
import wandb
from collections import Counter

# ---------------- Initialize W&B ---------------- #
wandb.init(project="Q1-weak-supervision-ner", name="conll2003-local-stats")

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
            # CoNLL: token, POS, chunk, ner_tag
            parts = line.split()
            token = parts[0]
            ner_tag = parts[-1]  # last column is the NER tag
            sentence.append(token)
            sentence_labels.append(ner_tag)
        # Add last sentence if file does not end with newline
        if sentence:
            sentences.append(sentence)
            labels.append(sentence_labels)
    return sentences, labels

# ---------------- Load Local Dataset ---------------- #
train_file = "D:\Assignment5_mlops\eng.train"
val_file = "D:\Assignment5_mlops\eng.testb"
test_file = "D:\Assignment5_mlops\eng.testa"

train_sentences, train_labels = load_conll_file(train_file)
val_sentences, val_labels = load_conll_file(val_file)
test_sentences, test_labels = load_conll_file(test_file)

# ---------------- Compute Dataset Statistics ---------------- #
num_train = len(train_sentences)
num_val = len(val_sentences)
num_test = len(test_sentences)

# Count entity occurrences
entity_counter = Counter()
for sent_labels in train_labels:
    entity_counter.update(sent_labels)

# Log dataset stats to W&B
wandb.summary["num_train_samples"] = num_train
wandb.summary["num_validation_samples"] = num_val
wandb.summary["num_test_samples"] = num_test
wandb.summary["entity_distribution"] = dict(entity_counter)

# ---------------- Print for Verification ---------------- #
print("Number of training samples:", num_train)
print("Number of validation samples:", num_val)
print("Number of test samples:", num_test)
print("Entity distribution:", dict(entity_counter))

wandb.finish()
