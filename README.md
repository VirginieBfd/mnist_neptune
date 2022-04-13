# MNIST Neptune

This is a test repo exploring integrating NeptuneAI into an ML project.

## Setup

Create environment:

```bash
conda create -n mnist-neptune python=3.9
pip install -r requirements.txt
```

Before running the script, create a `.env` file with:

- NEPTUNE_PROJECT_NAME: the Neptune project's name
- NEPTUNE_API_TOKEN: the Neptune project's API token

Run training script:

```bash
python src/train.py
```
