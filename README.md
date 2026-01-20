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

# Main Results
The foundation model was selected between BLIP‑2 and Qwen by comparing their zero-shot performance, ultimately choosing Qwen as the base model. The best-performing setup was validated using the Single‑Prompt and 1‑Img‑4‑Prompt strategies, evaluated with the Exact Match metric. Several configurations were explored by varying LoRA rank and dropout, while also applying a learning rate scheduler during training.

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

The table below reports the results of the best configuration for each strategy using R=16, D=0.05 and the cosine scheduler:
| Strategy   | Authors    | Genres  | Materials | Subjects 
|-----------------|-----------|---------|-----------|----------|
| SINGLE-PROMPT   |  17,43%   | **75,97%**  |  67,43%   |   6,66%  |
| 1IMG-4PROMPT    |  **18,28%**   | 75,68%  |  **69,59%**   |   **8,14%**  |

Among the evaluated approaches, the 1-IMG-4PROMPT strategy yields the best results.

Finally, the performance of the best-performing model (1-Img-4Prompt R=16 D=0.05 with cosine scheduler) was compared against the **zero-shot baseline** and the **multi-modal model**, which was treated as an upper bound. The table below reports the performance gains with respect to the **Exact Match (accuracy)** metric.

| Strategy   | Authors    | Genres  | Materials | Subjects 
|-----------------|-----------|---------|-----------|----------|
| ZERO-SHOT BASELINE   |  +3.28%   | +69,28  |  +47.79   |   +6,26%  |
| MULTI-MODEL    |  -0.17%   | -0.97%  |  -0,85%   |   +1.14%  |

# Folder Structure
```
.
├──Exam/
│   ├──adsp_notebook_artLM.ipynb/  # Main notebook with code, analysis, and results
│   ├──report.pdf/				   # Final project report and documentation
│   ├──checkpoints/			       # Presentation slides
├── Results/                       # Output directory for all results
│   ├── Outputs/                   # Generated plots and figures
│   	├── 1-IMG-4-PROMPT/		   # Results of IMG-4-PROMPT
│		├── MULTI-MODEL/		   # Results of MULTI-MODEL
│		├── SINGLE-PROMPT/         # Results of SINGLE-PROMPT                  
└── README.md					  # Project overview, instructions, and documentation
└── LINKS-ArtAI.pdf               # Kick off project presentation
```
# Installation
**Prerequisites**
- Python 3.10+
-	GPU with support to bfloat16  (ex: T4, L4, A100 su Colab)
-	PyTorch with CUDA
- Hugging Face Account: Access to Qwen2-VL-2B-Instruct.

**Setup**
1. Open adsp_notebook_artLM.ipynb in Jupyter or Colab.
2. Install the main libraries:
```bash
pip install -U transformers
!pip install qwen-vl-utils
!pip install peft
!pip install accelerate
!pip install sentencepiece
!pip install safetensors
```
3. In Colab, mount the Google Drive

```bash
from google.colab import drive
drive.mount('/content/drive')
```
4. Preparing Image Dataset
```bash
zip_path = "/content/drive/MyDrive/Dataset/foto.zip" #change the path
dest_path = "/content/dataset_veloce"
!unzip -q -o "{zip_path}" -d "{dest_path}"

folder = "/content/dataset_veloce/foto"
id_to_path = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder)}
```
5. Upload the model
```bash
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```
# Getting Started
## Training
	1.	Load the CSV files (train.csv and val.csv) and normalize any columns that contain stringified lists.
	2.	Perform exploratory data analysis on the metadata columns.
	3.	Run zero-shot evaluation using the Qwen2‑VL model.
	4.	Fine-tune the selected model with PEFT/LoRA.
	5.	Evaluate the fine-tuned models using the chosen metrics.

## Inference
Once a fine-tuned checkpoint is available, the model can be used to predict metadata for new artworks:
1. Load the base Qwen2‑VL model and attach the corresponding LoRA adapters (authors, materials, genres, or subjects).
2. Build the image–ID to file–path mapping (`id_to_path`) and prepare a DataFrame with the `images` column.
3. For each target category, call the appropriate inference utility (e.g., `run_inference_on_category`) to generate predictions.
4. Optionally, save the outputs to CSV and compute the same metrics used during validation (e.g., Exact Match).

# Limitations
- Rare labels: Poor performance on sparse subjects due to long-tail distribution
- Computational cost: High overhead for training/inference limits scalability
- Multi-label complexity: Challenges with co-occurring attributes

# Research Paper

This implementation is described in detail in the following paper:

> **“Automated Artwork Metadata Generation using Multimodal LLM”**  
> *Andò S., Baldi F., Cozzone R.* (2025)

---

## People

**Authors**  
- Simone Andò  
- Federico Baldi  
- Roberto Cozzone  

**Collaboration**  
- LINKS Foundation

---

# License

This project is intended **for academic and research use only**.


