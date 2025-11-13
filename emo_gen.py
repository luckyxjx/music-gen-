"""
emotion_cond_gen_multi_play_v2.py
Stage-3: Emotion-conditioned MIDI generator with multi-sample confidence stats + auto playback
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

 
# CONFIG
 
DATASET_PATH = "./EMOPIA_1.0"
LABELS_CSV = "label.csv"
NUM_FILES = 500
MAX_EVENTS = 200
SEQ_LEN = 32
BATCH_SIZE = 64
NUM_SAMPLES_PER_Q = 5  # generate 5 samples per quadrant
SEED = 42

OUTPUT_DIR = Path("emo_output")
OUTPUT_DIR.mkdir(exist_ok=True)

 
# MVP imports
 
from mvp_demo import (
    SimpleREMITokenizer,
    extract_monophonic_notes,
    pad_sequences,
    notes_to_midi,
    LSTMClassifier
)
from play_midi import play_midi

tokenizer = SimpleREMITokenizer()

 
# Dataset builder
 
def build_dataset(emopia_dir, labels_csv, tokenizer, num_files=NUM_FILES):
    labels_path = Path(emopia_dir) / labels_csv
    df = pd.read_csv(labels_path).sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows = df.iloc[:num_files]
    X_tokens, y_labels, file_paths = [], [], []

    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="Loading MIDI subset"):
        fname = r["ID"] + ".mid"
        quadrant = int(r["4Q"])
        midi_path = Path(emopia_dir) / "midis" / fname
        if not midi_path.exists():
            continue
        notes = extract_monophonic_notes(str(midi_path))
        toks = tokenizer.encode_monophonic(notes, max_events=MAX_EVENTS)
        if len(toks) < 4:
            continue
        X_tokens.append(toks)
        y_labels.append(quadrant - 1) 
        file_paths.append(str(midi_path))
    return X_tokens, y_labels, file_paths

 
# Datasets
 

#this dataset is for the classifier.
# It maps a full sequence to a single quadrant label.
class ClassifierDataset(Dataset):
    def __init__(self, X_tokens, y_quadrants, max_len=MAX_EVENTS):
        self.X = pad_sequences(X_tokens, max_len)
        self.y = np.array(y_quadrants, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]]).squeeze()
        )

# This dataset is for the GENERATOR.
# It maps a partial sequence to the next token.
class CondGenDataset(Dataset):
    def __init__(self, X_tokens, y_quadrants, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.X_in, self.y_out, self.conds = [], [], []

        for seq, q in zip(X_tokens, y_quadrants):
            for i in range(1, len(seq)):
                start = max(0, i - seq_len)
                inp = seq[start:i]
                if len(inp) == 0:
                    continue
                self.X_in.append(inp)
                self.y_out.append(seq[i])
                self.conds.append(q)

        self.X_in = pad_sequences(self.X_in, seq_len)
        self.y_out = np.array(self.y_out, dtype=np.int64)
        self.conds = np.array(self.conds, dtype=np.int64)

    def __len__(self):
        return len(self.X_in)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.X_in[idx]),
            torch.LongTensor([self.y_out[idx]]).squeeze(),
            torch.LongTensor([self.conds[idx]]).squeeze()
        )

 
# Conditional generator model
 
class LSTMCondGenerator(nn.Module):
    def __init__(self, vocab_size, num_quadrants=4, emb_dim=128, hidden_dim=256, cond_dim=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cond_emb = nn.Embedding(num_quadrants, cond_dim)
        self.lstm = nn.LSTM(emb_dim + cond_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, cond):
        token_emb = self.token_emb(x)
        cond_emb = self.cond_emb(cond).unsqueeze(1).expand(-1, x.size(1), -1)
        emb = torch.cat([token_emb, cond_emb], dim=-1)
        out, _ = self.lstm(emb)
        return self.fc(out)

 
# Classifier wrapper
 
class MVPClassifier:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def score(self, midi_path):
        notes = extract_monophonic_notes(midi_path)
        toks = self.tokenizer.encode_monophonic(notes, max_events=MAX_EVENTS)
        if not toks:
            return {'argmax': None, 'confidences': [0]*4}
        padded = pad_sequences([toks], MAX_EVENTS)
        x = torch.LongTensor(padded).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return {'argmax': int(np.argmax(probs)), 'confidences': probs.tolist()}

 
# Training functions
 
def train_cond_generator(model, dataloader, device, epochs=5, lr=1e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb, cb in dataloader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad()
            logits = model(xb, cb)
            last_logits = logits[:, -1, :]
            loss = loss_fn(last_logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[CondGen] Epoch {ep+1}/{epochs} loss={total_loss/len(dataloader):.4f}")

 
# Conditional sampling
 
def sample_conditional(model, seed_tokens, quadrant, length=128, temperature=1.0, top_k=10, device='cpu'):
    model.to(device)
    model.eval()
    seq = list(seed_tokens)
    for _ in range(length - len(seed_tokens)):
        inp = torch.LongTensor([seq[-SEQ_LEN:]]).to(device)
        cond = torch.LongTensor([quadrant]).to(device)
        with torch.no_grad():
            logits = model(inp, cond)
        logits = logits[0, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_val = values[-1]
            logits = torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        next_id = np.random.choice(len(probs), p=probs/probs.sum())
        seq.append(int(next_id))
    return seq

 
# Main
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

     
    # Dataset
     
    X_tokens, y_quadrants, _ = build_dataset(DATASET_PATH, LABELS_CSV, tokenizer, num_files=NUM_FILES)
    if not X_tokens:
        print("No MIDI data found. Check DATASET_PATH and LABELS_CSV.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X_tokens, y_quadrants, test_size=0.2, random_state=SEED, stratify=y_quadrants
    )

     
    # Train Stage-1 classifier
    clf_ds = ClassifierDataset(X_train, y_train, max_len=MAX_EVENTS)
    clf_loader = DataLoader(clf_ds, batch_size=BATCH_SIZE, shuffle=True)

    max_token_id = max([max(seq) for seq in X_tokens if seq])
    vocab_size = max_token_id + 1

    clf_model = LSTMClassifier(vocab_size=vocab_size, num_classes=4)
    clf_model.to(device)
    opt = torch.optim.AdamW(clf_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(3):
        clf_model.train()
        total_loss=correct=total=0
        #The loader now only yields two items (input, label)
        for xb, yb in clf_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = clf_model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        print(f"[Classifier] Epoch {ep+1}/3 loss={total_loss/len(clf_loader):.4f} acc={correct/total:.4f}")

    classifier = MVPClassifier(clf_model, tokenizer, device=device)

     
    # Train conditional generator
     
    #Use CondGenDataset here, which is correct for this task
    train_ds = CondGenDataset(X_train, y_train, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cond_gen = LSTMCondGenerator(vocab_size=vocab_size, num_quadrants=4)
    print("Training conditional generator...")
    train_cond_generator(cond_gen, train_loader, device, epochs=5)

     
    # Multi-sample generation per quadrant + playback
     
    summary_rows = []

    for q in range(4):
        confidences_q = []
        print(f"\nGenerating and playing samples for Quadrant Q{q+1}...")
        for s_idx in range(NUM_SAMPLES_PER_Q):
            seed = X_train[random.randint(0, len(X_train)-1)][:SEQ_LEN]
            sample_tokens = sample_conditional(cond_gen, seed, quadrant=q, length=120, device=device)
            notes = tokenizer.decode_to_notes(sample_tokens)
            out_file = OUTPUT_DIR / f"generated_cond_Q{q+1}_{s_idx+1}.mid"
            notes_to_midi(notes, str(out_file))

            # Automatic playback
            try:
                play_midi(str(out_file))
            except Exception as e:
                print(f"⚠️ Playback failed: {e}")

            clf_res = classifier.score(str(out_file))
            conf_target = clf_res['confidences'][q]
            confidences_q.append(conf_target)

        summary_rows.append({
            "Quadrant": f"Q{q+1}",
            "Min Confidence": f"{min(confidences_q):.2f}",
            "Avg Confidence": f"{np.mean(confidences_q):.2f}",
            "Max Confidence": f"{max(confidences_q):.2f}"
        })

    # Print summary
    print("\n=== Multi-Sample Conditional Generation Summary ===")
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()