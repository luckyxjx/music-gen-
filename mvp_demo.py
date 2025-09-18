"""
mvp_demo.py

"""

import os
import random
import math
from pathlib import Path
from collections import Counter
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from play_midi import play_midi  # 🔊 playback util

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "./EMOPIA_1.0"   # <-- set to your local EMOPIA path
LABELS_CSV = "label.csv"
NUM_FILES = 200
MAX_EVENTS = 200
SEQ_LEN = 32
GEN_EPOCHS = 3
CLF_EPOCHS = 3
BATCH_SIZE = 64
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Quantization
BPM = 120.0
Q16_SEC = 60.0 / BPM / 4.0
MAX_SHIFT_STEPS = 32

# Velocity buckets
VEL_BUCKETS = [0, 40, 80, 128]

# -----------------------------
# TOKENIZER
# -----------------------------
class SimpleREMITokenizer:
    def __init__(self, max_shift=MAX_SHIFT_STEPS, vel_buckets=VEL_BUCKETS):
        self.max_shift = max_shift
        self.vel_buckets = vel_buckets
        self.build_vocab()

    def build_vocab(self):
        self.token_to_id = {}
        self.id_to_token = {}
        tid = 0
        for s in range(1, self.max_shift + 1):
            t = f"TIME_SHIFT_{s}"
            self.token_to_id[t] = tid
            self.id_to_token[tid] = t
            tid += 1
        self.num_vel = len(self.vel_buckets) - 1
        for v in range(self.num_vel):
            t = f"VELOCITY_{v}"
            self.token_to_id[t] = tid
            self.id_to_token[tid] = t
            tid += 1
        for p in range(128):
            t = f"NOTE_ON_{p}"
            self.token_to_id[t] = tid
            self.id_to_token[tid] = t
            tid += 1
        self.vocab_size = tid

    def velocity_bucket(self, vel):
        for i in range(len(self.vel_buckets)-1):
            if self.vel_buckets[i] <= vel < self.vel_buckets[i+1]:
                return i
        return len(self.vel_buckets)-2

    def encode_monophonic(self, notes, max_events=MAX_EVENTS):
        if len(notes) == 0:
            return []
        notes = sorted(notes, key=lambda x: x[0])
        tokens, last_step = [], 0
        for (start, pitch, vel) in notes:
            step = int(round(start / Q16_SEC))
            delta = step - last_step
            while delta > 0:
                s = min(delta, self.max_shift)
                tokens.append(self.token_to_id[f"TIME_SHIFT_{s}"])
                delta -= s
            last_step = step
            vbucket = self.velocity_bucket(int(vel))
            tokens.append(self.token_to_id[f"VELOCITY_{vbucket}"])
            tokens.append(self.token_to_id[f"NOTE_ON_{int(pitch)}"])
            if len(tokens) >= max_events:
                break
        return tokens[:max_events]

    def decode_to_notes(self, token_ids, default_duration=0.5):
        cur_time, cur_vel = 0.0, 80
        notes = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, None)
            if token is None:
                continue
            if token.startswith("TIME_SHIFT_"):
                s = int(token.split("_")[-1])
                cur_time += s * Q16_SEC
            elif token.startswith("VELOCITY_"):
                vb = int(token.split("_")[-1])
                lo = self.vel_buckets[vb]
                hi = self.vel_buckets[vb+1] - 1
                cur_vel = int((lo + hi) / 2)
            elif token.startswith("NOTE_ON_"):
                p = int(token.split("_")[-1])
                notes.append((cur_time, p, cur_vel))
        return notes

# -----------------------------
# MIDI helpers
# -----------------------------
def extract_monophonic_notes(midi_path, max_notes=MAX_EVENTS):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        instrument = next((i for i in pm.instruments if not i.is_drum), None)
        if instrument is None:
            return []
        notes = [(n.start, n.pitch, n.velocity) for n in instrument.notes]
        notes.sort(key=lambda x: x[0])
        return notes[:max_notes]
    except Exception as e:
        print("Error reading MIDI:", midi_path, e)
        return []

def notes_to_midi(notes, out_path, instrument_program=0, note_duration=0.5):
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=instrument_program)
    for (start, pitch, vel) in notes:
        end = start + note_duration
        instr.notes.append(pretty_midi.Note(velocity=int(vel), pitch=int(pitch),
                                            start=float(start), end=float(end)))
    pm.instruments.append(instr)
    pm.write(out_path)

# -----------------------------
# Dataset building
# -----------------------------
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
        y_labels.append(quadrant)
        file_paths.append(str(midi_path))
    return X_tokens, y_labels, file_paths

# -----------------------------
# Collate helpers
# -----------------------------
def pad_sequences(seqs, maxlen, pad_value=0):
    padded = np.full((len(seqs), maxlen), pad_value, dtype=np.int64)
    for i, s in enumerate(seqs):
        trunc = s[:maxlen]
        padded[i, :len(trunc)] = np.array(trunc, dtype=np.int64)
    return padded

class TokenSeqDataset(Dataset):
    def __init__(self, sequences, labels, max_len=MAX_EVENTS):
        self.x = pad_sequences(sequences, max_len)
        self.y = np.array(labels, dtype=np.int64) - 1
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor([self.y[idx]]).squeeze()

# -----------------------------
# Models
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        emb = self.emb(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        return self.fc(out)

# -----------------------------
# Generator training prep
# -----------------------------
def make_generator_pairs(token_seqs, window=SEQ_LEN):
    X_in, y_out = [], []
    for seq in token_seqs:
        for i in range(1, len(seq)):
            start = max(0, i - window)
            inp = seq[start:i]
            X_in.append(inp)
            y_out.append(seq[i])
    return X_in, y_out

class GenDataset(Dataset):
    def __init__(self, X_in, y_out, max_len=SEQ_LEN):
        self.X = pad_sequences(X_in, max_len)
        self.y = np.array(y_out, dtype=np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.LongTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()

# -----------------------------
# Training helpers
# -----------------------------
def train_classifier(model, dataloader, device, epochs=CLF_EPOCHS, lr=1e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); total_loss=correct=total=0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        print(f"[Classifier] Epoch {ep+1}/{epochs} loss={total_loss/len(dataloader):.4f} acc={correct/total:.4f}")

def train_generator(model, dataloader, device, epochs=GEN_EPOCHS, lr=1e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); total_loss=0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            last_logits = logits[:, -1, :]
            loss = loss_fn(last_logits, yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[Generator] Epoch {ep+1}/{epochs} loss={total_loss/len(dataloader):.4f}")

# -----------------------------
# Sampling
# -----------------------------
def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

def sample_from_model(model, seed_tokens, length=128, temperature=1.0, top_k=10, device='cpu'):
    model.to(device).eval()
    seq = list(seed_tokens)
    for _ in range(length - len(seed_tokens)):
        inp = torch.LongTensor([seq[-SEQ_LEN:]]).to(device)
        with torch.no_grad():
            logits = model(inp)
        logits = logits[0, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            logits = top_k_logits(logits.unsqueeze(0), top_k)[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        next_id = np.random.choice(len(probs), p=probs/probs.sum())
        seq.append(int(next_id))
    return seq

# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    tokenizer = SimpleREMITokenizer()

    print("Building dataset (tokens + labels)...")
    X_tokens, y_labels, file_paths = build_dataset(DATASET_PATH, LABELS_CSV, tokenizer, num_files=NUM_FILES)
    if len(X_tokens) < 20:
        print("Not enough MIDI samples found. Make sure DATASET_PATH is correct and contains the EMOPIA dataset.")
        return

    # classifier
    X_train, X_val, y_train, y_val = train_test_split(X_tokens, y_labels, test_size=0.2, random_state=SEED, stratify=y_labels)
    clf_train_loader = DataLoader(TokenSeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    clf = LSTMClassifier(tokenizer.vocab_size)
    print("Training classifier...")
    train_classifier(clf, clf_train_loader, device)

    # generator
    gen_X_in, gen_y = make_generator_pairs(X_train)
    if not gen_X_in:
        print("No generator training pairs found."); return
    gen_loader = DataLoader(GenDataset(gen_X_in, gen_y), batch_size=BATCH_SIZE, shuffle=True)
    gen = LSTMGenerator(tokenizer.vocab_size)
    print("Training generator...")
    train_generator(gen, gen_loader, device)

    # generation
    print("Sampling raw melodies...")
    seeds = [s[:min(len(s), SEQ_LEN)] for s in random.sample(X_tokens, k=min(5, len(X_tokens)))]
    generated_files = []
    for i, seed in enumerate(seeds[:3]):
        print(f"  Generating sample {i+1}...")
        sample_tokens = sample_from_model(gen, seed, length=120, device=device)
        # 🚨 ensure audible note
        if not any("NOTE_ON" in tokenizer.id_to_token[t] for t in sample_tokens):
            print("⚠️ No NOTE_ON found, injecting fallback Middle C...")
            sample_tokens = [
                tokenizer.token_to_id["VELOCITY_2"],
                tokenizer.token_to_id["NOTE_ON_60"],
                tokenizer.token_to_id["TIME_SHIFT_4"]
            ]
        notes = tokenizer.decode_to_notes(sample_tokens)
        out_file = OUTPUT_DIR / f"generated_uncond_{i+1}.mid"
        notes_to_midi(notes, str(out_file))
        generated_files.append(str(out_file))
        print("    Saved:", out_file)
        try: play_midi(str(out_file))
        except Exception as e: print(f"⚠️ Playback failed: {e}")

    # classify generated
    print("\nClassifying generated samples...")
    clf.to(device).eval()
    for gf in generated_files:
        notes = extract_monophonic_notes(gf)
        toks = tokenizer.encode_monophonic(notes, max_events=MAX_EVENTS)
        if not toks: continue
        padded = pad_sequences([toks], MAX_EVENTS)
        x = torch.LongTensor(padded).to(device)
        with torch.no_grad(): logits = clf(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred, conf = int(np.argmax(probs))+1, float(np.max(probs))
        print(f"  {os.path.basename(gf)} -> Quadrant {pred}, Conf {conf:.3f}")

    print("\nMVP run complete. Files in:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
