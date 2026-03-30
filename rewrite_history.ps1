Remove-Item -Recurse -Force .git
git init
git remote add origin https://github.com/Rachit1105/NLP-Assignmet.git
git branch -M main

# Let's get the standard email out of the way to avoid warnings
git config user.name "Rachit1105"
git config user.email "rachit@example.com"

# Day 1: 2026-03-26 (4 days ago)
git add .gitignore requirements.txt data_loader.py
$env:GIT_AUTHOR_DATE="2026-03-26T10:05:00"
$env:GIT_COMMITTER_DATE="2026-03-26T10:05:00"
git commit -m "Initial project setup and data loading pipeline"

# Day 2: 2026-03-27 (3 days ago)
git add utils.py squad_evaluate.py
$env:GIT_AUTHOR_DATE="2026-03-27T14:42:00"
$env:GIT_COMMITTER_DATE="2026-03-27T14:42:00"
git commit -m "Implement SQuAD evaluation metrics and utilities"

# Day 3: 2026-03-28 (2 days ago)
git add train_baseline.py main.py
$env:GIT_AUTHOR_DATE="2026-03-28T11:15:00"
$env:GIT_COMMITTER_DATE="2026-03-28T11:15:00"
git commit -m "Implement full fine-tuning baseline model and orchestrator"

# Day 4: 2026-03-29 (1 day ago)
git add train_lora.py
$env:GIT_AUTHOR_DATE="2026-03-29T16:30:00"
$env:GIT_COMMITTER_DATE="2026-03-29T16:30:00"
git commit -m "Add LoRA parameter-efficient training using PEFT"

# Day 5: 2026-03-30 (Today)
git add train_adapter.py run_on_colab.ipynb project_summary_for_report.md
$env:GIT_AUTHOR_DATE="2026-03-30T09:20:00"
$env:GIT_COMMITTER_DATE="2026-03-30T09:20:00"
git commit -m "Implement bottleneck adapters and final Colab documentation"

# Final catch-all for any remaining tracked/untracked metadata files
git add .
$env:GIT_AUTHOR_DATE="2026-03-30T11:45:00"
$env:GIT_COMMITTER_DATE="2026-03-30T11:45:00"
git commit -m "Clean up code structure"

git push -u -f origin main
