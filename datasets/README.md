# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below if you
need to re-fetch them.

## Dataset 1: QReCC (Question Rewriting in Conversational Context)

### Overview
- **Source**: HuggingFace dataset `svakulenk0/qrecc`
- **Size**: train 63,501; test 16,451
- **Format**: JSON array (`qrecc-training.json`, `qrecc-test.json`)
- **Task**: conversational question rewriting / intent preservation
- **Splits**: train, test
- **License**: refer to dataset card

### Download Instructions

**Direct download (used here):**
```bash
mkdir -p datasets/qrecc
curl -L -o datasets/qrecc/qrecc-training.json \
  https://huggingface.co/datasets/svakulenk0/qrecc/resolve/main/qrecc-training.json
curl -L -o datasets/qrecc/qrecc-test.json \
  https://huggingface.co/datasets/svakulenk0/qrecc/resolve/main/qrecc-test.json
```

### Loading the Dataset
```python
import json
with open("datasets/qrecc/qrecc-training.json", "r", encoding="utf-8") as f:
    train = json.load(f)
```

### Sample Data
See `datasets/qrecc/samples/samples.json`.

### Notes
- The HF dataset card contains files with mixed columns; direct file download
  avoids schema casting errors when using `datasets`.

## Dataset 2: CANARD-QuReTeC Gold Supervision

### Overview
- **Source**: HuggingFace dataset `uva-irlab/canard_quretec`
- **Size**: train 20,181; dev 2,196; test 3,373
- **Format**: JSON list (`*_gold_supervision.json`)
- **Task**: term-level query resolution / conversational search
- **Splits**: train, dev, test
- **License**: refer to dataset card

### Download Instructions
```bash
mkdir -p datasets/canard_quretec
curl -L -o datasets/canard_quretec/train_gold_supervision.json \
  https://huggingface.co/datasets/uva-irlab/canard_quretec/resolve/main/train_gold_supervision.json
curl -L -o datasets/canard_quretec/dev_gold_supervision.json \
  https://huggingface.co/datasets/uva-irlab/canard_quretec/resolve/main/dev_gold_supervision.json
curl -L -o datasets/canard_quretec/test_gold_supervision.json \
  https://huggingface.co/datasets/uva-irlab/canard_quretec/resolve/main/test_gold_supervision.json
```

### Loading the Dataset
```python
import json
with open("datasets/canard_quretec/train_gold_supervision.json", "r", encoding="utf-8") as f:
    train = json.load(f)
```

### Sample Data
See `datasets/canard_quretec/samples/`.

### Notes
- The HF dataset uses a custom loading script; direct downloads avoid loader
  compatibility issues.
