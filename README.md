# Transformer-based Text Generation Model

This project implements a transformer-based text generation model trained on the Cosmopedia-100k dataset. The model is designed to generate coherent and contextually relevant text based on a given prompt.

## Dataset

- **Source**: HuggingFaceTB/cosmopedia-100k
- **Size**: 100,000 samples
- **Content**: Diverse text data covering various topics

### Dataset Structure

The dataset is split into training and test sets:
- Training set: 80% of the data
- Test set: 20% of the data

Data preprocessing includes:
- Removal of unnecessary columns
- Text cleaning (lowercasing, special character removal)
- Tokenization using SentencePiece

## Model Architecture

The model is a "Decoder-Only" style Transformer with the following specifications:

- Embedding size: 256
- Number of transformer blocks: 8
- Number of attention heads: 8
- Vocabulary size: 8000 tokens (SentencePiece tokenizer)
- Positional encoding: Sinusoidal
- Total parameters: Approximately 20 million

## Data Preprocessing

1. Text cleaning (lowercasing, special character removal)
2. Tokenization using SentencePiece
3. Padding sequences to a fixed length
4. Converting tokens to tensor format

## Tokenization

- **Tokenizer**: SentencePiece
- **Vocabulary size**: 8000
- **Special tokens**: 
  - PAD: 0
  - BOS: 1
  - EOS: 2

## Training

- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Loss function**: Cross-Entropy Loss
- **Batch size**: 256
- **Number of epochs**: 150
- **Mixed precision training**: FP16

## Accuracy and Evaluation

The model's performance is evaluated using the training loss curve:

- Starting loss: 6.0461
- Final loss: 2.9447

The decreasing loss indicates that the model is learning to generate more accurate predictions over time.

![Training Loss Curve](training_loss_curve.png)

## Usage

To generate text using the trained model:

```python
sample_text = generate_text(
    tf_generator, sp, 
    start_text="Effects of online dating: ", 
    max_length=100, 
    temperature=0.2,
    top_p=0.4,
    repetition_penalty=1.5,
    interval=30
)
print(sample_text)
```

Parameters:
- `temperature`: Controls randomness (lower values for more deterministic output)
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Reduces word repetition
- `interval`: Frequency of topic reinforcement

The model uses nucleus sampling and repetition penalty to improve the quality and diversity of generated text.
