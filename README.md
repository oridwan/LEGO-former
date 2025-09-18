# LEGO-former: Crystal Structure Generation & Analysis Toolkit

This repository provides tools for the generation, transformation, and analysis of crystal structures

---

## Installation

### 1. Install Pyxtal (with Julia support)
```bash
export INSTALL_JULIA=1 && pip install pyxtal
```
To check if the installation is successful, run:
```python
from juliacall import Main as jl
jl.seval("using CrystalNets")
print("Success")
```

### 2. GULP Installation
- Download the GULP package
- Extract and compile:
```bash
tar -zvxf gulp-6.2-2.tgz
cd gulp-6.2/Src
./mkgulp
```
- Add GULP to your environment (edit `~/.bashrc` or `~/.zshrc`):
```bash
export PATH=/path/to/gulp-6.2/Src:$PATH
export GULP_LIB=/path/to/gulp-6.2/Libraries
```
- Test GULP:
```bash
gulp < gulp.in
```
You should see a timing analysis and `Job Finished` message at the end.

### 3. MACE (Machine Learning Force Fields)
```bash
pip install --upgrade pip
pip install mace-torch
```

---

## How to Run a Job
1. Put your generated data in the `data/sample` folder.
2. Submit a relaxation job using SLURM:
```bash
sbatch -J <filename_without_.csv> util/slurm/run_relax.sh
```
**Example:**
For `data/sample/Transformer-class_token.csv`:
```bash
sbatch -J Transformer-class_token util/slurm/run_relax.sh
```

---

## Quick Reference
- **Pyxtal**: Structure generation and manipulation
- **GULP**: Energy and force field calculations
- **MACE**: Machine learning force fields
- **SLURM**: Batch job submission for structure relaxation

---

For more details, see the documentation in each subfolder or script.


