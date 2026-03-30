# Project Summary: BERT Question Answering with Parameter-Efficient Fine-Tuning

**Goal:** Solve the inefficiency of full fine-tuning of Large Language Models (LLMs) by implementing and comparing Parameter-Efficient Fine-Tuning (PEFT) methods on a Question Answering task.

**Base Model:** `bert-base-uncased` (HuggingFace)
**Dataset:** SQuAD v1.1 subset (Stanford Question Answering Dataset)

---

## 1. Project Architecture & Setup
The project was structured into an end-to-end Machine Learning pipeline written in Pytorch and HuggingFace Transformers.

**Core files generated:**
- `data_loader.py`: Handles downloading SQuAD, advanced tokenization, and torch DataLoader generation.
- `squad_evaluate.py`: Contains the logic for converting model logits to text spans and calculating Extractive QA metrics.
- `train_baseline.py`: Implements full-model fine-tuning.
- `train_lora.py`: Implements Low-Rank Adaptation (LoRA) using the `peft` library.
- `train_adapter.py`: Implements custom bottleneck Adapter layers injected into the BERT architecture.
- `main.py`: The orchestrator script that manages the training loop comparisons via CLI arguments.
- `utils.py`: Shared utilities for model parameter counting, timing, and result visualization.

---

## 2. Dataset Processing (SQuAD)
Because BERT has a maximum sequence length (384 tokens), Wikipedia context passages in SQuAD are often too long to fit into a single pass. 

**Techniques implemented:**
- **Sliding Window Context Handling:** We implemented a `doc_stride` of 128. If a context is too long, the tokenizer splits it into multiple overlapping chunks.
- **Offset Mapping:** We mapped the token indices back to the original character positions in the raw text so we can accurately slice the exact answer strings from the original paragraph during evaluation.
- **Answer Targeting:** If an answer spans across the chunk boundary, the model is trained to point to the `[CLS]` token (index 0), indicating "No answer exists in this chunk."

---

## 3. The Three Models Implemented

### Model A: 100% Full Fine-Tuning (The Baseline)
- Loaded `AutoModelForQuestionAnswering` from `bert-base-uncased`.
- No frozen layers.
- **Parameters Trained:** ~108.8 Million (100% of the model).
- **Optimizer:** AdamW with a linear warmup scheduler (10% warmup steps).

### Model B: Low-Rank Adaptation (LoRA)
- Used the HuggingFace `peft` library.
- Froze all base BERT weights.
- Injected trainable rank decomposition matrices (Rank = 8, Alpha = 16) into the `query` and `value` projection layers of the BERT Self-Attention blocks.
- **Parameters Trained:** ~296,000 (0.27% of the model).

### Model C: Bottleneck Adapters (Custom Implementation)
- We built a custom PyTorch class (`BertLayerWithAdapter`) that wraps the standard HuggingFace `BertLayer`.
- Froze all base BERT weights.
- Inserted two custom bottleneck adapter blocks per layer:
    1. One after the Self-Attention output.
    2. One after the Feed-Forward Network (FFN) output.
- **Architecture per Adapter:** `LayerNorm -> Linear(768 down to 64) -> GELU -> Linear(64 up to 768) -> Residual Connection`.
- Initialized adapter weights to near-zero (`std=1e-3`) so their initial presence does not disrupt the pre-trained BERT representations.
- Unfroze the final QA output classification head.
- **Parameters Trained:** ~2.4 Million (2.17% of the model).

---

## 4. Evaluation & Metrics
To evaluate the models, we implemented a custom prediction extractor. Since BERT outputs `start_logits` and `end_logits` for every token, we:
1. Extract the top 20 combination pairs of start and end tokens (N-best extraction).
2. Filter out invalid pairs (where end token is before start token, or length exceeds 30 words).
3. Use the `offset_mapping` to slice the highest-scoring valid token span from the raw context paragraph.
4. Pass these strings to the official HuggingFace `evaluate("squad")` library.

**Metrics Tracked:**
- **Exact Match (EM) Accuracy:** Percentage of predictions that matched the ground truth perfectly, character for character.
- **F1 Score:** Measures the overlap of words, giving partial credit.
- **Training Time:** Recorded via a custom built `Timer` context manager to showcase the speed efficiency of PEFT.
- **Trainable Parameters:** Calculated the footprint ratio to demonstrate memory efficiency.

---

## 5. Google Colab & Operational Work
Finally, we structured a Jupyter Notebook (`run_on_colab.ipynb`) designed specifically for remote GPU execution constraint-handling. 
- It clones the repository via Git.
- Dynamically separates the runs into different isolated notebook cells so a crash in an earlier run won't wipe out subsequent metric comparisons.
- Features an automated Google Drive backup script (`shutil.copytree`) that maps the `saved_models` and `results` JSON footprints to the user's permanent `My Drive > NLP` cloud staging area.
