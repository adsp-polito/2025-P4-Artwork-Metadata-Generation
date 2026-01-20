# Automated Artwork Metadata Generation using MLLM - ADSP Project

This repository contains **ARTLM**, an end-to-end framework for **automated artwork metadata generation** using **Multimodal Large Language Models (MLLMs)**.

The project was developed as part of the **Applied Data Science Project** at **Politecnico di Torino (2025/2026)** and is described in detail in the accompanying research paper.

The system leverages a **vision–language model fine-tuned with Parameter-Efficient Fine-Tuning (PEFT)** techniques to generate **structured, multi-label metadata** for artworks directly from images.

# Project Motivation
The large-scale digitization of cultural heritage collections has created an urgent need for **scalable and consistent metadata generation**. Manual annotation is costly, time-consuming, and does not scale to collections containing thousands of artworks.

This project addresses these challenges by:
- Framing metadata extraction as a **generative, multi-task, multi-label problem**
- Using a **single multimodal model** to predict heterogeneous metadata fields (authors, genres, materials, subjects)
- Reducing computational costs through **Low-Rank Adaptation (LoRA)**

The approach is designed to be **scalable, flexible, and deployable in resource-constrained environments**, while maintaining strong semantic accuracy.

<img width="975" height="281" alt="Screenshot 2026-01-09 alle 19 19 30" src="https://github.com/user-attachments/assets/a24719fe-e342-455d-84c0-bd74d5ee882f" />

# Key features
- **Multimodal Metadata Generation**  
  Generates structured metadata directly from artwork images using Foundation Model QWEN2-VL-INSTRUCT

- **Multi-Label & Open-Vocabulary Output**  
  Supports multiple valid labels per category without relying on closed label sets.

- **Unified Generative Framework**  
  A single model predicts all metadata categories without task-specific heads.

- **Parameter-Efficient Fine-Tuning (PEFT)**  
  Uses **LoRA** adapters to fine-tune large models efficiently.

- **Flexible Prompting Strategies**
  - Single Prompt (all categories together)
  - 1 Image – 4 Prompts (category-specific inference)
  - Multi-Model upper-bound comparison

- **Robust Evaluation**
  - Exact Match (EM)
  - Semantic Similarity (cosine similarity with MPNet embeddings)
  - BLEU and ROUGE-1 scores

# How it works
The framework formulates metadata extraction as a **conditional multimodal generation task**:
**Core Steps:**
1. **Vision Encoder**  
   Extracts visual tokens from artwork images.

2. **Textual Category Prompts**  
   Each metadata category is encoded as a textual prompt.

3. **Multimodal Alignment**  
   Visual and textual tokens are aligned into a shared representation.

4. **Autoregressive Decoder**  
   Generates metadata sequences token-by-token.

# Dataset
The dataset is constructed from **Wikidata**, an open-source structured knowledge base.

**Dataset Statistics:**
- **9,008 artworks**
  - 7,252 training samples
  - 1,756 validation samples
- High-resolution artwork images
- Strongly **imbalanced multi-label distributions**, especially in *Subjects*
<img width="793" height="202" alt="Screenshot 2026-01-20 alle 09 10 28" src="https://github.com/user-attachments/assets/ca01ab42-010f-4568-931f-fd63ec7742b8" />
<img width="1252" height="277" alt="Screenshot 2026-01-20 alle 09 15 07" src="https://github.com/user-attachments/assets/122804ff-81dc-4bb4-88c6-743eca4d1cf7" />

The dataset enables realistic evaluation of large-scale cultural heritage metadata generation.

# Folder Structure
```
.
├── adsp_notebook_artLM.ipynb   # Main notebook with code, analysis, and results
├── model_results/              # Output directory for all results
│   ├── plots/                  # Generated plots and figures
│   └── results_tables/         # Quantitative evaluation tables
├── report/                     # Final project report and documentation
├── Checkpoints/                # Presentation slides
└── README.md                   # Project overview, instructions, and documentation
```
# How to Use
1. Open adsp_notebook_artLM.ipynb in Jupyter, VSCode or Colab.
2. Place Dataset_ArtAI.csv in the same folder as the notebook.
3 Run all cells in order. Results will be saved in model_results/ and displayed in the notebook.

#Main Results
The experiments used to validate the best-performing model were conducted using the **Single-Prompt** and **1Img4Prompt** strategies.  
Multiple configurations were evaluated by varying the **LoRA rank** and **dropout values**, while applying a **learning rate scheduler** during training.

The table below reports a comparative analysis of the two prompting strategies for different Rank and Dropout configuration:
| SINGLE-PROMPT   | Authors   | Genres | Materials | Subjects 
|-----------------|-----------|---------|----------|----------|
| R = 8,D=0.1     |  **17,26%**   | 75,34%  |  66,57%  |   6,66%  |
| R = 16,D=0.05   |  17,06%   | **75,57%**  |  **66,86%**  |   **7,00%**  |

| 1IMG-4PROMPT   | Authors    | Genres  | Materials | Subjects 
|-----------------|-----------|---------|-----------|----------|
| R = 8,D=0.1     |  17,71%   | 75,80%  |  68,74%   |   **7,92%**  |
| R = 16,D=0.05   |  **18,45%**   | **76,37%**  |  **69,19%**   |   7,74%  |

In both strategies, the best-performing configuration uses a **LoRA rank of 16** and a **dropout value of 0.05**.  
For both strategies, the application of a **learning rate scheduler** leads to performance improvements.

The table below reports the best configuration for each strategy using R=16, D=0.05 and the cosine scheduler:
| Strategy   | Authors    | Genres  | Materials | Subjects 
|-----------------|-----------|---------|-----------|----------|
| SINGLE-PROMPT   |  17,43%   | **75,97%**  |  67,43%   |   6,66%  |
| 1IMG-4PROMPT    |  **18,28%**   | 75,68%  |  **69,59%**   |   **8,14%**  |

Among the evaluated approaches, the 1IMG-4PROMPT strategy yields the best results.

