#  Advanced SMS Spam Detection with DistilBERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%20Transformers-4.57.0-ffca28?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-76b900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-00d4aa?style=for-the-badge)

![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-99.28%25-brightgreen?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/F1%20Score-97.35%25-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-5,572%20SMS-blue?style=for-the-badge)

*A state-of-the-art SMS spam detection system using DistilBERT with advanced preprocessing and hyperparameter optimization.*

[ Documentation](#-documentation)  [ Quick Start](#-quick-start)  [ Performance](#-performance)  [ Contributing](#-contributing)

</div>

---

##  Table of Contents
- [ Features](#-features)
- [ Quick Start](#-quick-start)
- [ Performance Highlights](#-performance-highlights)
- [ Technical Details](#-technical-details)
- [ Usage Examples](#-usage-examples)
- [ Documentation](#-documentation)
- [ Contributing](#-contributing)
- [ License](#-license)

##  Features

###  Advanced Preprocessing
- Custom SMS text normalization and cleaning
- Intelligent abbreviation handling (u  you, ur  your)
- URL, email, and phone pattern detection
- Emoji processing and standardization
- Smart stop word filtering

###  DistilBERT Architecture
- Fine-tuned transformer model (40% smaller, 97% performance)
- Selective layer unfreezing for efficiency
- Attention visualization capabilities
- Mobile-optimized design

###  Analysis & Training
- Comprehensive data exploration
- Advanced hyperparameter optimization with Optuna
- Early stopping and model checkpointing
- Detailed performance metrics and visualizations

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/fahiiim/Advanced-SMS-SPAM-Classification-STI-Client-Project.git
cd Advanced-SMS-SPAM-Classification-STI-Client-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Usage

```python
# Example usage
from spam_classifier import SMSSpamClassifier, predict_spam

# Load model
model = SMSSpamClassifier.load_pretrained()

# Predict
text = "Your message here"
result, confidence = predict_spam(text, model)
print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

##  Performance Highlights

| Metric | Score |
|--------|-------|
| Accuracy | 99.28% |
| Precision | 96.49% |
| Recall | 98.21% |
| F1 Score | 97.35% |

##  Technical Details

### Model Architecture
- Base: DistilBERT (distilbert-base-uncased)
- Parameters: ~66M (vs 110M BERT-base)
- Hidden Size: 768
- Attention Heads: 12
- Layers: 6

### Training Configuration
- Batch Size: 16/32 (train/eval)
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Warmup Steps: 10%
- Gradient Clipping: 1.0

##  Usage Examples

### Basic Classification

```python
# Single message classification
text = "URGENT! You`ve won 1000! Text WIN to 81010 now!"
result, confidence = model.predict(text)
print(f"Classification: {result} (Confidence: {confidence:.2%})")
```

### Batch Processing

```python
# Process multiple messages
messages = [
    "Hey, are we meeting tomorrow?",
    "URGENT! You`ve won 1000!",
    "Your package will arrive today"
]

results = model.predict_batch(messages)
for msg, (result, conf) in zip(messages, results):
    print(f"{result}: {msg[:40]}... ({conf:.1%})")
```

##  Documentation

Detailed documentation is available in the `docs` folder:
- [Complete API Reference](docs/api_reference.md)
- [Advanced Usage Guide](docs/advanced_usage.md)
- [Model Architecture](docs/architecture.md)
- [Training Details](docs/training.md)

##  Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with  by Fahim
<br>
 Contact: [GitHub](https://github.com/fahiiim)
</div>
