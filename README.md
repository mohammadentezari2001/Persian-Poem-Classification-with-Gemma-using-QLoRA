# Persian Poem Classification with Gemma (using QLoRA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)

This project implements a fine-tuning pipeline for the **Google Gemma (270M)** Large Language Model to classify Persian poems by their poets.

The benchmark focuses on Parameter-Efficient Fine-Tuning (PEFT) methods, specifically comparing **4-bit QLoRA** against **8-bit Quantized LoRA** with varying adapter ranks ($r=8$ and $r=16$).

## Results

Based on a sample size of 5000 rows over 5 epochs, the following results were observed:

| Method | Rank (r) | Trainable Params | Time (sec) | Max Memory (GB) | Test F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QLoRA (4-bit)** | 8 | 750,080 | **1469.5** | **1.20** | 0.864 |
| **QLoRA (4-bit)** | 16 | 1,487,360 | 1472.0 | 1.69 | 0.870 |
| **Quantized (8-bit)** | 8 | 750,080 | 2094.9 | 1.74 | **0.896** |
| **Quantized (8-bit)** | 16 | 1,487,360 | 2623.3 | 1.78 | 0.872 |

**Key Findings:**
*   **Efficiency:** QLoRA 4-bit (r=8) was the fastest and most memory-efficient method (1.2 GB VRAM).
*   **Performance:** Standard 8-bit Quantization (r=8) achieved the highest F1 score (0.896), though it required roughly 42% more training time than the 4-bit variant.

##  Features

*   **Model**: Fine-tunes `google/gemma-3-270m`.
*   **Quantization Support**:
    *   **QLoRA (4-bit)**: Uses `bitsandbytes` for 4-bit normal float quantization.
    *   **8-bit Quantization**: Uses standard 8-bit loading.
*   **Text Processing**: Includes custom handling for **Persian/Farsi text** (reshaping and BiDi support) to ensure correct rendering in plots.
*   **Benchmarking**: Automatically runs experiments across different configurations and outputs a summary CSV.
*   **Visualization**: Generates confusion matrices for each experimental run.

##  Dataset Structure

The script expects the dataset to be placed in a folder named `Dataset` in the project root. The CSV files must contain at least two columns: `text` (the poem) and `poet` (the label).

```text
├── main.ipynb
├── requirements.txt
├── Dataset/
│   ├── Train.csv
│   ├── Test.csv
│   └── Val.csv
└── ...
```

##  Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mohammadentezari2001/Persian-Poem-Classification-with-Gemma-using-QLoRA-.git
    cd Persian-Poem-Classification-with-Gemma-using-QLoRA
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Required libraries include: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `pandas`, `scikit-learn`, `matplotlib`, `arabic-reshaper`, `python-bidi`.*

##  Configuration

Hyperparameters are centralized in the `CONFIG` dictionary within the script. Default settings used for the benchmark above:

```python
CONFIG = {
    "model_name": "google/gemma-3-270m",
    "sample_size": 5000,    # 5000 samples used for speed optimization
    "epochs": 5,
    "batch_size": 8,
    "lr": 2e-4,
    "r": [8, 16],           # LoRA ranks tested
}
```

##  Usage

Run the main notebook/script to start the benchmarking process:

```bash
python main.ipynb
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

*   Hugging Face for the `transformers` and `peft` libraries.
*   Google DeepMind for the Gemma model.
*   `arabic_reshaper` and `python-bidi` for solving RTL text rendering issues in Matplotlib.
