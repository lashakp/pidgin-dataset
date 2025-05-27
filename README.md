# 🧹 Pidgin Cleaner

This project cleans and analyzes a Pidgin English dataset sourced from the Hugging Face Hub. It removes empty or invalid rows, normalizes text by removing punctuation/special characters, and exports the cleaned dataset.

## 📂 Project Structure

- `pidgin_cleaning.py` – Script to clean and analyze the dataset
- `data/` – Folder containing the cleaned dataset
- `requirements.txt` – Python packages needed
- `.gitignore` – Ignore large files/folders like `data/`

## 🚀 How to Run

```bash
pip install -r requirements.txt
python pidgin_cleaning.py
