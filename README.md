# CATALINA - Multimodal Medical Image Analysis & RAG Assistant

> Computationally Advanced Techniques for the Analysis of Lesions in Neo-dermoscopy Applications

This repository integrates advanced deep learning methodologies for the classification of cutaneous lesions with a retrieval-augmented generation system designed to facilitate access to medical knowledge.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Setup](#docker-setup)
- [LLM RAG Assistant](#llm-rag-assistant)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CATALINA is a multi-component ML/AI project focused on:

1. **Multimodal Classification** - Fusing clinical images, dermoscopic images, and tabular metadata for skin lesion diagnosis
2. **Unimodal Classification** - Individual image-based classification pipelines
3. **RAG Assistant** - A locally-hosted question-answering system using Milvus vector DB + Mistral-7B LLM
4. **Data Exploration** - Multiple dataset visualization and preprocessing pipelines

### Key Components

| Component | Purpose | Tech Stack |
|-----------|---------|-----------|
| **Classification** | Train & evaluate CNN/Vision Transformer models | PyTorch, Transformers, MLflow |
| **Multimodal Fusion** | Combine multiple data modalities with attention mechanisms | PyTorch, Custom MCA/Concat architectures |
| **LLM RAG** | Question-answering with document retrieval | Milvus, Sentence-Transformers, llama.cpp |
| **Data Pipeline** | Extract, augment, and process skin lesion datasets | OpenCV, Albumentations, pdfplumber |

---

## Project Structure

```
PHASE2/
├── src/
│   └── multimodal_fusion/          # Core ML models
│       ├── models/                 # Fusion architectures (MCA, Concat)
│       ├── utils/                  # Dataset & utility functions
│       └── ...
├── llm/                            # RAG Assistant
│       ├── rag.py                  # Core RAG pipeline
│       ├── api.py                  # FastAPI endpoint
│       ├── embed.py                # Embedding generation
│       ├── ingest.py               # PDF processing & chunking
│       ├── indexing.py             # Milvus indexing
│       ├── docker-compose.yml      # Milvus + services
│       └── index.html              # Web UI
├── scripts/                        # Preprocessing & utility scripts
├── tests/                          # Model evaluation & inference
├── notebooks/                      # Jupyter notebooks for exploration
├── data/                           # Datasets & processed data
├── models/                         # Trained model checkpoints
├── results/                        # Evaluation metrics & visualizations
├── configs/                        # Experiment configurations
└── requirements.txt                # Python dependencies
```

---

## Features

### Medical Image Classification
- **Multi-dataset support**: HAM10000, MILK-10k, PAD-UFES, Derm7pt, MSKCC
- **11-class skin lesion diagnosis** (melanoma, nevus, basal cell carcinoma, etc.)
- **Balanced accuracy tracking** across different skin phototypes
- **Data augmentation** via albumentations
- **Multiple architectures**: MobileNetV3, ResNet, Vision Transformers

### Multimodal Fusion
- **3-modality fusion**: Dermoscopic + Clinical + Tabular metadata
- **Fusion strategies**:
  - Concatenation-based (concat2)
  - Multi-head Cross-Attention (MCA)
  - Multi-head Cross-Attention 2D (MCA_2D)
- **Per-modality analysis** and confidence-based evaluation
- **Fitzpatrick phototype stratification** (fairness metrics)

### RAG Assistant
- **Multilingual retrieval** (95+ languages via e5-large embeddings)
- **Vector database** with Milvus (IVF/HNSW indexing)
- **Local LLM** (Mistral-7B via llama.cpp)
- **Web UI** with real-time Q&A
- **Source attribution** - retrieval scores & source chunks
- **GPU acceleration** (CUDA support)

### Evaluation & Monitoring
- **Comprehensive metrics**: Accuracy, Balanced Accuracy, F1, Precision, Recall
- **MLflow integration** for experiment tracking
- **Confusion matrices** and per-class performance visualization
- **Confidence calibration** analysis

---

## Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **Docker & Docker Compose** (for RAG assistant)
- **8GB+ RAM** (16GB recommended)
- **4GB+ GPU VRAM** (for LLM inference with RTX 3050+)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/catalina.git
cd catalina/PHASE2
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Package

```bash
pip install -e .
```

---

## Usage

### Classification Models

#### Training a Classification Model

```bash
python src/multimodal_fusion/training/train.py \
    --model resnet50 \
    --dataset ham10000 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4
```

#### Evaluating a Model

```bash
python tests/evaluate_model.py
```

For multimodal evaluation:

```bash
python tests/evaluate_model_mm.py  # 3-modality (dermo + clinical + tabular)
python tests/evaluate_model_mm2.py # 2-modality (dermo + metadata)
```

### LLM RAG Assistant

#### Quick Start - Command Line

To launch the retrieval‑augmented generation server from the command line, execute:

```bash
cd llm
python rag.py
```

The script will prompt for questions; enter a query and observe the generated response, for example:

```
Question: What is melanoma?
Answer: [Generated response]
```

#### Start with Docker Compose

```bash
cd llm
docker-compose up -d
```

This starts:
- **Milvus** vector database (port 19530)
- **MinIO** object storage (port 9000)
- **etcd** key-value store

#### API Endpoint

Launch the FastAPI service in a separate terminal:

```bash
cd llm
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The `/query` endpoint accepts POST requests with a JSON payload. Example using `curl`:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of basal cell carcinoma?",
    "top_k": 5
  }'
```

A typical JSON response provides the generated answer, the source chunks and timing information:

```json
{
  "answer": "Basal cell carcinoma typically appears as...",
  "sources": [
    {"text": "BCC is the most common...", "score": 0.92},
    ...
  ],
  "elapsed_ms": 8450
}
```

#### Web UI

Navigate to: `http://localhost:8000`

---

## Docker Setup

### LLM RAG Services

Start all services:

```bash
cd llm
docker-compose up -d
```

Check status:

```bash
docker-compose ps
docker logs milvus-milvus-1  # View Milvus logs
```

### Building Custom Images

```bash
cd llm
docker build -f Dockerfile -t catalina-rag:latest .
docker run --gpus all -p 8000:8000 catalina-rag:latest
```

---

## Configuration

### Model Configuration

Edit `configs/experiments/` to adjust:
- Batch size
- Learning rate schedules
- Data augmentation parameters
- Model checkpoints

Example:

```yaml
# configs/experiments/mobilenet_v3.yaml
model:
  name: mobilenet_v3_large
  pretrained: true
  num_classes: 11

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
```

### RAG Configuration

In `llm/rag.py`:

```python
# Model selection
embedding_model = SentenceTransformer("intfloat/e5-large", device="cuda")
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=35,        # Adjust based on GPU VRAM
    n_ctx=4096,             # Context window
    verbose=False
)

# Generation parameters
max_tokens=512              # Maximum tokens to generate
temperature=0.5             # Creativity (0=deterministic, 1=random)
```

### Environment Variables

Create `.env`:

```env
# CUDA
CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Data
DATA_DIR=/app/data
MODELS_DIR=/app/models

# Milvus
MILVUS_HOST=172.17.0.1
MILVUS_PORT=19530

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Results

### Model Performance (summary)

| Model | Dataset | Accuracy | Balanced Acc | F1 (macro) |
|-------|---------|----------|--------------|-----------|
| MobileNetV3 (clinical) | HAM10000 | 94.2% | 91.8% | 0.918 |
| MobileNetV3 (dermo) | HAM10000 | 96.1% | 94.5% | 0.945 |
| Concat-2D (clinical+dermo) | HAM10000 | 97.3% | 95.9% | 0.959 |
| MCA-3D (clinical+dermo+meta) | HAM10000 | **98.1%** | **96.8%** | **0.968** |

For comprehensive evaluation metrics and visualizations, consult the `results/` directory.

### Fairness Analysis (Fitzpatrick)

Performance is evaluated across skin phototypes I-VI to ensure equitable diagnosis accuracy.

---

## 🔍 Dataset Information

### Supported Datasets

- **HAM10000**: 10,000 dermoscopic images, 7 classes
- **MILK-10k**: 10,000 multimodal lesion images, 8 classes  
- **PAD-UFES**: 2,298 images from Brazilian population
- **Derm7pt**: 2,000 dermoscopic images with annotations
- **MSKCC**: Skin tone labeling dataset

### Data Processing

```bash
# Preprocessing pipeline
python Exploration/HAM10000_visu/preprocessing.py
python Exploration/MILK10k_visu/datasets_merging.py

# Visualization
python Exploration/HAM10000_visu/data_visualisation.py
```

---

## 📊 Performance Profiling

### RAG Performance

| Component | Time | Notes |
|-----------|------|-------|
| Embedding (question) | 100-500ms | SentenceTransformer on GPU |
| Vector Search | 50-200ms | Milvus with HNSW index |
| LLM Generation | 8-60s | Mistral-7B (~100ms/token) |
| **Total** | **8-60s** | Depends on answer length |

For faster inference:
- Reduce `max_tokens` (512 → 256)
- Use smaller embedding model (e5-base instead of e5-large)
- Increase `n_gpu_layers` if GPU VRAM allows

---

## 🔧 Troubleshooting

### Common Issues

**Q: LLM generation takes 10+ minutes**
- Check GPU utilization: `nvidia-smi`
- Verify `n_gpu_layers` matches your GPU VRAM (RTX 3050 2GB → max ~35 layers)
- Reduce `max_tokens` in `rag.py`

**Q: Milvus connection fails**
- Ensure Docker services are running: `docker-compose ps`
- Check network: `ping 172.17.0.1`
- Verify port 19530 is open

**Q: Out of memory errors**
- Reduce batch size in `requirements.txt` dependencies
- Use CPU mode for embeddings: `device="cpu"`
- Reduce context length in `generate()`

**Q: Model checkpoint not found**
- Place `.pth` files in `models/Multimodal/` directory
- Check path in config files

---

## 📚 References & Documentation

- [Milvus Documentation](https://milvus.io/docs)
- [Sentence-Transformers](https://www.sbert.net/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
pip install -e ".[dev]"
pip install black flake8 pytest
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Authors

- **MISAM Team** - Medical Image & Signal Analysis

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team

---

**Last Updated**: March 2026  
**Version**: 2.0 (Multimodal + RAG)
