# Low-Dimensional Subspace Results

This subrepository contains code to generate Figures 4, 5, 7 (and their 2D counterparts).

## Requirements

- Python 3.13+
- CUDA-compatible GPU (recommended for faster computation with JAX)

## Setup

### Option 1: Using pip (Recommended)

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using conda

1. Create a conda environment:
```bash
conda create -n low-dim-subspace python=3.13
```

2. Activate the environment:
```bash
conda activate low-dim-subspace
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Figures

The repository includes scripts to generate figures from the experiments. All figure scripts are located in the `scripts/` directory.

### Available Figures

- **Figure 4**: Matrix factorization experiments (30×30 symmetric matrix)
- **Figure 5**: LoRA-BERT experiments
- **Figure 7**: Deep LoRA subspace analysis

### Running Figure Scripts

To generate a specific figure, run:

```bash
python scripts/figure_4.py
```

```bash
python scripts/figure_5.py
```

```bash
python scripts/figure_7.py
```

Each script will:
1. Load or generate the required data
2. Train models as needed
3. Generate both 2D and 3D visualizations
4. Save PDF outputs to the project root directory

### Output Files

The scripts generate PDF files in the project root:
- `figure_4_2d.pdf` and `figure_4_3d.pdf`
- `figure_5_2d.pdf` and `figure_5_3d.pdf`
- `figure_7_2d.pdf`, `figure_7a_3d.pdf`, `figure_7b_3d.pdf`, `figure_7c_3d.pdf`

### Expected Runtime

Training and figure generation may take several minutes to hours depending on your hardware. The scripts use JAX, which will automatically utilize GPU acceleration if available.