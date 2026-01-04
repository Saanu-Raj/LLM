# 🩺 TinyLlama Medical Assistant

A fine-tuned TinyLlama 1.1B model specialized in allopathic medicine, trained using LoRA (Low-Rank Adaptation) on a custom medical dataset.

## 🎯 Features

- **Fine-tuned Model**: Specialized knowledge of 10+ common medicines
- **LoRA Adaptation**: Efficient fine-tuning with only 2.2M trainable parameters
- **4-bit Quantization**: Memory-efficient inference
- **User Authentication**: Role-based access (admin, doctor, student)
- **Medical Disclaimer**: Safety warnings on all responses
- **Interactive UI**: Clean Streamlit interface with adjustable parameters

## 📊 Model Details

- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (r=8, alpha=16)
- **Training Data**: 500 medical Q&A pairs
- **Training Accuracy**: 97.83%
- **Medicines Covered**: Paracetamol, Ibuprofen, Amoxicillin, Metformin, Atorvastatin, Amlodipine, Omeprazole, Cetirizine, Azithromycin, Losartan

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-assistant.git
cd medical-assistant

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Login Credentials

```
Username: admin     Password: admin123
Username: doctor    Password: doc123
Username: student   Password: student123
```

## 📁 Project Structure

```
medical-assistant/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── tinyllama-medical-lora/        # Fine-tuned model weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
└── README.md
```

## 💡 Example Queries

- "What is Paracetamol used for?"
- "Tell me about Ibuprofen"
- "What is Metformin?"
- "Uses of Amoxicillin"
- "What is Atorvastatin for?"

## ⚙️ Model Parameters

Adjust these in the sidebar:
- **Temperature** (0.1-1.5): Controls randomness
- **Max Tokens** (32-256): Response length
- **Top-p** (0.1-1.0): Nucleus sampling

## ⚠️ Medical Disclaimer

This AI assistant is for educational purposes only. Always consult a qualified healthcare provider for medical advice.

## 🔧 Technical Stack

- **Framework**: Streamlit
- **Model**: TinyLlama 1.1B + LoRA
- **Libraries**: Transformers, PEFT, BitsAndBytes, PyTorch
- **Quantization**: 4-bit NF4

## 📄 License

MIT License

## 👥 Authors

Your Name - Medical AI Research

## 🙏 Acknowledgments

- TinyLlama team for the base model
- Hugging Face for transformers library
- PEFT library for LoRA implementation
