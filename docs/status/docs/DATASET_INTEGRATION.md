# Dataset Integration Guide

This guide explains how to integrate additional emotion-labeled MIDI datasets beyond EMOPIA.

## Supported Dataset Formats

The system supports multiple dataset formats:

1. **EMOPIA** - Primary dataset with quadrant-based emotion labels
2. **Lakh MIDI** - Large MIDI dataset with emotion annotations
3. **MAESTRO** - Classical piano dataset with emotion labels
4. **Generic JSON** - Custom datasets with JSON metadata

## Quick Start

### 1. Prepare Your Dataset

If you have a directory of MIDI files without metadata:

```bash
python scripts/prepare_dataset.py create-template /path/to/midi/files -o metadata.json
```

This creates a `metadata.json` template. Edit it to add emotion labels:

```json
[
  {
    "path": "song1.mid",
    "emotion": "joy"
  },
  {
    "path": "song2.mid",
    "emotion": "sadness"
  }
]
```

### 2. Validate Your Dataset

```bash
python scripts/prepare_dataset.py validate /path/to/dataset
```

This checks:
- All MIDI files exist and are valid
- Emotion labels are correct
- Shows emotion distribution

### 3. Configure Additional Datasets

Edit your config file or code:

```python
from src.config import DataConfig

config = DataConfig(
    dataset_path="./EMOPIA_1.0",
    additional_datasets=[
        # Simple path (auto-detects format)
        "./my_dataset",
        
        # Or specify type explicitly
        {"path": "./lakh_midi", "type": "lakh"},
        {"path": "./maestro", "type": "maestro"},
        {"path": "./custom_data", "type": "json"}
    ]
)
```

## Emotion Categories

All datasets are mapped to these 6 emotion categories:
- **joy** - Happy, joyful, excited
- **sadness** - Sad, melancholic, depressed
- **anger** - Angry, aggressive, furious
- **calm** - Calm, peaceful, relaxed
- **surprise** - Surprised, shocked
- **fear** - Scared, fearful, anxious

## Dataset Format Specifications

### Generic JSON Format

Create a `metadata.json` file in your dataset directory:

```json
[
  {
    "path": "relative/path/to/file.mid",
    "emotion": "joy"
  }
]
```

### Lakh MIDI Format

Expected structure:
```
lakh_midi/
├── metadata.json
└── midi_files/
    ├── song1.mid
    └── song2.mid
```

`metadata.json`:
```json
[
  {
    "path": "midi_files/song1.mid",
    "emotion": "happy"
  }
]
```

Emotion keywords are automatically mapped (e.g., "happy" → "joy").

### MAESTRO Format

Expected structure:
```
maestro/
├── maestro_emotions.csv
└── 2004/
    ├── song1.midi
    └── song2.midi
```

`maestro_emotions.csv`:
```csv
midi_filename,emotion
2004/song1.midi,joyful
2004/song2.midi,melancholic
```

## CLI Tools

### Scan Directory

Find all MIDI files in a directory:

```bash
python scripts/prepare_dataset.py scan /path/to/directory
```

### Create Template

Generate metadata template:

```bash
# JSON format (default)
python scripts/prepare_dataset.py create-template /path/to/midi -o metadata.json

# CSV format
python scripts/prepare_dataset.py create-template /path/to/midi -o metadata.csv -f csv

# With default emotion
python scripts/prepare_dataset.py create-template /path/to/midi -e joy
```

### Validate Dataset

Check dataset integrity:

```bash
python scripts/prepare_dataset.py validate /path/to/dataset
```

Output shows:
- Total files vs valid files
- Missing files
- Invalid MIDI files
- Emotion distribution

### Merge Datasets

Combine multiple datasets:

```bash
python scripts/prepare_dataset.py merge dataset1/ dataset2/ dataset3/ -o merged_metadata.json
```

## Example: Adding a Custom Dataset

1. **Organize your MIDI files:**
```
my_music/
├── happy_songs/
│   ├── song1.mid
│   └── song2.mid
└── sad_songs/
    ├── song3.mid
    └── song4.mid
```

2. **Create metadata template:**
```bash
python scripts/prepare_dataset.py create-template my_music/ -o my_music/metadata.json
```

3. **Edit metadata.json:**
```json
[
  {"path": "happy_songs/song1.mid", "emotion": "joy"},
  {"path": "happy_songs/song2.mid", "emotion": "joy"},
  {"path": "sad_songs/song3.mid", "emotion": "sadness"},
  {"path": "sad_songs/song4.mid", "emotion": "sadness"}
]
```

4. **Validate:**
```bash
python scripts/prepare_dataset.py validate my_music/
```

5. **Add to config:**
```python
config = DataConfig(
    dataset_path="./EMOPIA_1.0",
    additional_datasets=["./my_music"]
)
```

## Programmatic Usage

### Load Multiple Datasets

```python
from src.dataset import EMOPIADataset
from src.config import DataConfig

config = DataConfig(
    dataset_path="./EMOPIA_1.0",
    additional_datasets=[
        "./lakh_midi",
        {"path": "./maestro", "type": "maestro"}
    ]
)

dataset = EMOPIADataset(config, split="train")
print(f"Total samples: {len(dataset)}")
```

### Use Dataset Loaders Directly

```python
from src.dataset_loaders import get_dataset_loader

# Auto-detect format
loader = get_dataset_loader("./my_dataset")
samples = loader.load()

# Specify format
loader = get_dataset_loader("./my_dataset", dataset_type="json")
samples = loader.load()
```

### Validate Dataset

```python
from src.utils.dataset_utils import validate_dataset_metadata, print_validation_report

results = validate_dataset_metadata("./my_dataset")
print_validation_report(results)

if results["valid"]:
    print("Dataset is ready to use!")
```

## Emotion Mapping

The system automatically maps various emotion keywords to the 6 core categories:

| Input Keywords | Mapped To |
|---------------|-----------|
| happy, joyful, excited | joy |
| sad, melancholic, depressed | sadness |
| angry, aggressive, furious | anger |
| calm, peaceful, relaxed | calm |
| surprised, shocked | surprise |
| scared, fearful, anxious | fear |

## Troubleshooting

### "Metadata file not found"
- Ensure `metadata.json` exists in the dataset directory
- Check the file name matches (case-sensitive)

### "Invalid MIDI file"
- MIDI file may be corrupted
- File may have no notes
- Use validation tool to identify problematic files

### "Emotion not recognized"
- Use one of the 6 core emotions
- Or use a keyword that maps to them
- Check spelling

### Dataset not loading
- Verify paths are correct (absolute or relative to working directory)
- Check file permissions
- Run validation tool first

## Best Practices

1. **Always validate** datasets before training
2. **Use consistent emotion labels** across datasets
3. **Balance emotion classes** for better training
4. **Keep metadata files** in the dataset directory
5. **Use relative paths** in metadata for portability
6. **Document your datasets** with README files
7. **Version control** metadata files

## Advanced: Custom Dataset Loader

Create a custom loader for your format:

```python
from src.dataset_loaders import DatasetLoader

class MyCustomLoader(DatasetLoader):
    def load(self):
        samples = []
        # Your custom loading logic
        for file in self.dataset_path.glob("*.mid"):
            samples.append({
                "path": str(file),
                "emotion": self._detect_emotion(file),
                "quadrant": None,
                "source": "custom"
            })
        return samples
    
    def _detect_emotion(self, file):
        # Your emotion detection logic
        return "calm"
```

Register it:

```python
from src.dataset_loaders import get_dataset_loader

# Monkey patch or modify get_dataset_loader function
```
