FROM nvcr.io/nvidia/pytorch:24.01-py3

# Git et Git LFS pour télécharger les modèles HF
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Hugging Face Hub + librairies LLM
RUN pip install --upgrade pip && pip install \
    huggingface_hub \
    transformers \
    accelerate \
    bitsandbytes \
    sentence-transformers

# TensorRT (optionnel pour tests GPU)
RUN pip install tensorrt tensorrt_llm --extra-index-url https://pypi.nvidia.com

# Répertoire pour stocker les modèles HF
RUN mkdir /models
WORKDIR /workspace
