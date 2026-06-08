# FYDP-Model Dataset Guide

## Table of Contents
1. [Current Cell Lines (6)](#current-cell-lines-6)
2. [Cell Line Descriptions & Differences](#cell-line-descriptions--differences)
3. [Additional Cell Lines (4+ Recommended)](#additional-cell-lines-4-recommended)
4. [Data Sources & Download Links](#data-sources--download-links)
5. [Processing Pipeline (RTX 5070 Ti Optimized)](#processing-pipeline-rtx-5070-ti-optimized)
6. [Step-by-Step Processing Guide](#step-by-step-processing-guide)
7. [Configuration Updates](#configuration-updates)

---

## Current Cell Lines (6)

| Cell Line | Tissue Type | Disease State | BENGI Files | Status |
|-----------|-------------|---------------|-------------|--------|
| **GM12878** | Lymphoblastoid (B-cell) | Normal | HiC, CTCF-ChIAPET, RNAPII-ChIAPET | Ready |
| **HeLa-S3** | Cervical epithelial | Cancer (adenocarcinoma) | HiC, CTCF-ChIAPET, RNAPII-ChIAPET | Ready |
| **K562** | Bone marrow | Cancer (CML leukemia) | HiC | Ready |
| **IMR90** | Lung fibroblast | Normal | HiC | Ready |
| **HMEC** | Mammary epithelial | Normal | HiC | Ready |
| **NHEK** | Keratinocyte (skin) | Normal | HiC | Ready |

### Current Training/Test Split
- **Training**: GM12878, HeLa, K562, IMR90 (4 cell lines)
- **Testing**: HMEC, NHEK (2 cell lines - completely unseen)

---

## Cell Line Descriptions & Differences

### 1. GM12878 (Lymphoblastoid B-Cell)
- **Origin**: Female, Caucasian, EBV-immortalized lymphoblastoid
- **Tissue**: Blood/Immune system
- **Characteristics**:
  - ENCODE Tier 1 cell line (most comprehensive data)
  - Represents hematopoietic lineage
  - High transcriptional activity in immune-related genes
- **EPI Relevance**: Captures immune-specific enhancer-promoter interactions
- **Sample Count**: ~56,244 EPI pairs (largest dataset)

### 2. HeLa-S3 (Cervical Cancer)
- **Origin**: Female, African American, cervical adenocarcinoma
- **Tissue**: Cervical epithelial
- **Characteristics**:
  - Most widely used cancer cell line
  - Highly proliferative, aneuploid genome
  - Aberrant gene regulation patterns
- **EPI Relevance**: Cancer-specific regulatory rewiring
- **Sample Count**: ~26,769 EPI pairs

### 3. K562 (Chronic Myelogenous Leukemia)
- **Origin**: Female, Caucasian, CML blast crisis
- **Tissue**: Bone marrow/Blood
- **Characteristics**:
  - ENCODE Tier 1 cell line
  - Erythroid/megakaryocyte differentiation potential
  - BCR-ABL fusion gene present
- **EPI Relevance**: Leukemia-specific enhancer usage
- **Sample Count**: ~82,983 EPI pairs

### 4. IMR90 (Lung Fibroblast)
- **Origin**: Female, Caucasian, fetal lung
- **Tissue**: Lung mesenchyme
- **Characteristics**:
  - Primary (non-immortalized) cell line
  - Limited passage lifespan
  - Normal diploid genome
- **EPI Relevance**: Normal developmental regulation
- **Sample Count**: ~21,431 EPI pairs

### 5. HMEC (Human Mammary Epithelial Cells)
- **Origin**: Female, breast tissue
- **Tissue**: Mammary epithelial
- **Characteristics**:
  - Primary cells from normal breast tissue
  - Relevant for breast cancer studies
  - Hormone-responsive enhancers
- **EPI Relevance**: Breast-specific regulatory landscape
- **Sample Count**: ~27,248 EPI pairs (TEST SET)

### 6. NHEK (Normal Human Epidermal Keratinocytes)
- **Origin**: Human skin epidermis
- **Tissue**: Skin/Epithelial
- **Characteristics**:
  - Primary cells from normal skin
  - High turnover, differentiation program
  - Skin-specific gene expression
- **EPI Relevance**: Epithelial differentiation enhancers
- **Sample Count**: ~22,431 EPI pairs (TEST SET)

---

## Additional Cell Lines (4+ Recommended)

### Priority 1: High Data Availability (ENCODE Tier 1/2)

| Cell Line | Tissue | Disease | BENGI Available | ENCODE Data | Priority |
|-----------|--------|---------|-----------------|-------------|----------|
| **HUVEC** | Endothelial (umbilical vein) | Normal | Yes (HiC) | Full | HIGH |
| **NHLF** | Lung fibroblast | Normal | Yes (limited) | Full | HIGH |
| **H1-hESC** | Embryonic stem cell | Normal | Limited | Full | HIGH |
| **HepG2** | Liver hepatocyte | Cancer (hepatoma) | Limited | Full | MEDIUM |

### Priority 2: Additional Cancer Types

| Cell Line | Tissue | Disease | BENGI Available | ENCODE Data | Priority |
|-----------|--------|---------|-----------------|-------------|----------|
| **A549** | Lung epithelial | Cancer (adenocarcinoma) | Limited | Good | MEDIUM |
| **MCF-7** | Breast epithelial | Cancer (adenocarcinoma) | No | Good | MEDIUM |
| **HCT116** | Colon epithelial | Cancer (colorectal) | No | Good | LOW |

### Priority 3: For GTEx/Tissue Studies

| Tissue | Source | BENGI Available | Notes |
|--------|--------|-----------------|-------|
| **Liver** | GTEx | Yes | Tissue-level, not cell line |
| **Pancreas** | GTEx | Yes | Tissue-level |
| **Thyroid** | GTEx | Yes | Tissue-level |

### Recommended 10 Cell Lines (Final List)

| # | Cell Line | Role | Data Status |
|---|-----------|------|-------------|
| 1 | GM12878 | Training | Ready |
| 2 | HeLa-S3 | Training | Ready |
| 3 | K562 | Training | Ready |
| 4 | IMR90 | Training | Ready |
| 5 | HMEC | Testing | Ready |
| 6 | NHEK | Testing | Ready |
| 7 | **HUVEC** | Training/Testing | **TO ADD** |
| 8 | **NHLF** | Training/Testing | **TO ADD** |
| 9 | **H1-hESC** | Training/Testing | **TO ADD** |
| 10 | **HepG2** | Training/Testing | **TO ADD** |

---

## Data Sources & Download Links

### A. BENGI Benchmark Files (EPI Pairs)

**Main Repository**: https://github.com/weng-lab/BENGI

```bash
# Clone BENGI repository
git clone https://github.com/weng-lab/BENGI.git

# Navigate to benchmark files
cd BENGI/Benchmark/All-Pairs.Natural-Ratio/
```

**Direct Download Links** (v3 benchmarks):
```
# Current Cell Lines (already have)
GM12878.HiC-Benchmark.v3.tsv.gz
GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz
GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz
HeLa.HiC-Benchmark.v3.tsv.gz
HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz
HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv.gz
K562.HiC-Benchmark.v3.tsv.gz
IMR90.HiC-Benchmark.v3.tsv.gz
HMEC.HiC-Benchmark.v3.tsv.gz
NHEK.HiC-Benchmark.v3.tsv.gz

# Additional Cell Lines to Download
HUVEC.HiC-Benchmark.v3.tsv.gz    # If available in BENGI
CD34.CHiC-Benchmark.v3.tsv.gz    # Alternative hematopoietic
```

### B. Epigenetic Signal Files (ENCODE)

**ENCODE Portal**: https://www.encodeproject.org/

**Required Marks per Cell Line** (8 tracks):
1. CTCF (narrowPeak format)
2. DNase-seq (bigWig format)
3. H3K27ac (bigWig format)
4. H3K27me3 (bigWig format)
5. H3K36me3 (bigWig format)
6. H3K4me1 (bigWig format)
7. H3K4me3 (bigWig format)
8. H3K9me3 (bigWig format)

**Search URLs by Cell Line**:

```bash
# HUVEC
https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name=HUVEC&assay_title=DNase-seq&assay_title=Histone+ChIP-seq&assay_title=TF+ChIP-seq&target.label=CTCF

# NHLF
https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name=NHLF&assay_title=DNase-seq&assay_title=Histone+ChIP-seq

# H1-hESC
https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name=H1&assay_title=DNase-seq&assay_title=Histone+ChIP-seq

# HepG2
https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name=HepG2&assay_title=DNase-seq&assay_title=Histone+ChIP-seq
```

### C. Reference Genome

**hg19 Reference** (required for DNA sequence extraction):
```bash
# Download from UCSC
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz

# Create index for pyfaidx
pip install pyfaidx
python -c "import pyfaidx; pyfaidx.Fasta('hg19.fa')"
```

---

## Processing Pipeline (RTX 5070 Ti Optimized)

### Hardware Specifications
- **GPU**: NVIDIA RTX 5070 Ti
- **VRAM**: 16GB GDDR7
- **CUDA Cores**: ~8,960
- **Recommended Batch Size**: 64-128 (with 16GB VRAM)

### Software Requirements

```bash
# Create conda environment
conda create -n fydp python=3.10 -y
conda activate fydp

# PyTorch with CUDA 12.x (for RTX 50 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Data processing
pip install pyBigWig pyfaidx numpy pandas tqdm pyyaml scikit-learn

# Visualization
pip install matplotlib seaborn
```

### Memory Optimization for 5070 Ti

```python
# In train.py or config.yaml
config = {
    'data': {
        'batch_size': 64,  # Safe for 16GB VRAM
        # Increase to 128 if memory allows after testing
    },
    'training': {
        'num_workers': 4,  # Optimal for disk I/O
        'pin_memory': True,  # Faster GPU transfer
    }
}

# Enable mixed precision (FP16) for 2x speedup
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Step-by-Step Processing Guide

### Step 1: Download Raw Data

```bash
# Create directory structure
mkdir -p data/raw/{bigWig,narrowPeak,BENGI}
cd data/raw

# Example: Download HUVEC data from ENCODE
# 1. Go to ENCODE portal
# 2. Search for HUVEC + DNase-seq
# 3. Download bigWig file (signal p-value)

# Download naming convention:
# bigWig: {CellLine}_{Mark}.bigWig
# narrowPeak: {CellLine}_{Mark}.narrowPeak.gz
```

### Step 2: Convert bigWig to PyTorch Tensors

Create file: `scripts/process_bigwig.py`

```python
#!/usr/bin/env python3
"""
Convert bigWig files to PyTorch tensors for FYDP-Model.
Optimized for RTX 5070 Ti with batch processing.
"""

import os
import sys
import torch
import numpy as np
import pyBigWig
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Chromosome sizes (hg19)
CHROM_SIZES = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430,
    "chr4": 191154276, "chr5": 180915260, "chr6": 171115067,
    "chr7": 159138663, "chr8": 146364022, "chr9": 141213431,
    "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
    "chr16": 90354753, "chr17": 81195210, "chr18": 78077248,
    "chr19": 59128983, "chr20": 63025520, "chr21": 48129895,
    "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
}

BIN_SIZE = 500  # 500bp bins

def process_bigwig(bw_path, output_path, cell_line, mark):
    """Process a single bigWig file to .pt tensor."""
    print(f"Processing: {cell_line} - {mark}")

    bw = pyBigWig.open(bw_path)
    result = {}

    for chrom, size in tqdm(CHROM_SIZES.items(), desc=f"{cell_line}_{mark}"):
        num_bins = size // BIN_SIZE
        values = np.zeros(num_bins, dtype=np.float32)

        for i in range(num_bins):
            start = i * BIN_SIZE
            end = min((i + 1) * BIN_SIZE, size)
            try:
                val = bw.stats(chrom, start, end, type="mean")
                if val and val[0] is not None:
                    values[i] = val[0]
            except:
                pass

        result[chrom] = torch.from_numpy(values)

    bw.close()

    # Save as PyTorch tensor
    torch.save(result, output_path)
    print(f"Saved: {output_path}")
    return output_path

def process_narrowpeak(peak_path, output_path, cell_line, mark="CTCF"):
    """Process narrowPeak to binary .pt tensor (0/1 per bin)."""
    import gzip

    print(f"Processing: {cell_line} - {mark} (narrowPeak)")
    result = {chrom: torch.zeros(size // BIN_SIZE) for chrom, size in CHROM_SIZES.items()}

    opener = gzip.open if peak_path.endswith('.gz') else open
    with opener(peak_path, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom not in result:
                continue

            start_bin = start // BIN_SIZE
            end_bin = min(end // BIN_SIZE + 1, len(result[chrom]))
            result[chrom][start_bin:end_bin] = 1.0

    torch.save(result, output_path)
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Process all bigWig/narrowPeak files for a cell line."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True, help='Cell line name (e.g., HUVEC)')
    parser.add_argument('--input-dir', default='./data/raw', help='Input directory')
    parser.add_argument('--output-dir', default='./data/genomic_data/processed', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    marks_bigwig = ['DNase', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
    marks_narrowpeak = ['CTCF']

    # Process bigWig files
    for mark in marks_bigwig:
        input_file = os.path.join(args.input_dir, 'bigWig', f'{args.cell}_{mark}.bigWig')
        if os.path.exists(input_file):
            output_file = os.path.join(args.output_dir, f'bigWig_{args.cell}_{mark}.500bp.pt')
            process_bigwig(input_file, output_file, args.cell, mark)
        else:
            print(f"WARNING: {input_file} not found")

    # Process narrowPeak files
    for mark in marks_narrowpeak:
        input_file = os.path.join(args.input_dir, 'narrowPeak', f'{args.cell}_{mark}.narrowPeak.gz')
        if os.path.exists(input_file):
            output_file = os.path.join(args.output_dir, f'narrowPeak_{args.cell}_{mark}.500bp.pt')
            process_narrowpeak(input_file, output_file, args.cell, mark)
        else:
            print(f"WARNING: {input_file} not found")


if __name__ == '__main__':
    main()
```

### Step 3: Run Processing

```bash
# Process each new cell line
python scripts/process_bigwig.py --cell HUVEC
python scripts/process_bigwig.py --cell NHLF
python scripts/process_bigwig.py --cell H1-hESC
python scripts/process_bigwig.py --cell HepG2
```

### Step 4: Update Configuration Files

#### Update `feats_config.json`:

```json
{
    "_location": "E:/AI Models/FYDP-Model/data/genomic_data/processed",
    "GM12878": { ... },
    "HeLa": { ... },
    "IMR90": { ... },
    "K562": { ... },
    "HMEC": { ... },
    "NHEK": { ... },
    "HUVEC": {
        "CTCF": "narrowPeak_HUVEC_CTCF.500bp.pt",
        "DNase": "bigWig_HUVEC_DNase.500bp.pt",
        "H3K27ac": "bigWig_HUVEC_H3K27ac.500bp.pt",
        "H3K27me3": "bigWig_HUVEC_H3K27me3.500bp.pt",
        "H3K36me3": "bigWig_HUVEC_H3K36me3.500bp.pt",
        "H3K4me1": "bigWig_HUVEC_H3K4me1.500bp.pt",
        "H3K4me3": "bigWig_HUVEC_H3K4me3.500bp.pt",
        "H3K9me3": "bigWig_HUVEC_H3K9me3.500bp.pt"
    },
    "NHLF": { ... },
    "H1": { ... },
    "HepG2": { ... }
}
```

### Step 5: Download BENGI Files for New Cell Lines

```bash
# If HUVEC.HiC-Benchmark.v3.tsv.gz exists in BENGI
# Copy to data/BENGI/ folder

# For cell lines not in BENGI, you can create custom benchmarks from:
# 1. Hi-C data (4D Nucleome, GEO)
# 2. ChIA-PET data (ENCODE)
# 3. Capture Hi-C data
```

### Step 6: Training with New Cell Lines

```bash
# Example: Train on 6 cell lines, test on 4
python train.py \
    --train-cells GM12878 HeLa K562 IMR90 HUVEC NHLF \
    --test-cells HMEC NHEK H1 HepG2 \
    --epochs 100 \
    --patience 15 \
    --batch-size 64
```

---

## Configuration Updates

### Updated `configs/config.yaml` (10 cell lines)

```yaml
experiment_name: "POCD_Kansformer_v7_10cells"

paths:
  save_dir: "./output_10cells"
  bengi_dir: "./data/BENGI"
  feats_config: "./data/genomic_data/CTCF_DNase_6histone_local.500.json"
  ref_genome: "../Loco EPI-main/hg19.fa"

data:
  sequence_length: 6000
  epigenetic_bins: 5000
  n_epigenetic_features: 9
  kmer_size: 3
  batch_size: 64  # Optimal for RTX 5070 Ti (16GB)
  bin_size: 500
  seq_len_bp: 2500000
  enhancer_window: 3000
  promoter_window: 3000
  encoder_fit_samples: 10000  # Increase for more cell lines
  feats_order:
    - "CTCF"
    - "DNase"
    - "H3K27ac"
    - "H3K27me3"
    - "H3K36me3"
    - "H3K4me1"
    - "H3K4me3"
    - "H3K9me3"

training:
  lr: 0.0001
  epochs: 150  # More epochs for larger dataset
  lambda_dist: 1.0
  grad_clip: 1.0
  patience: 20  # More patience with diverse data
  num_workers: 4  # Increase for faster loading
  weight_decay: 0.01  # Add regularization for larger model
  use_cosine_scheduler: true  # Better for longer training
```

---

## Data Availability Summary

### ENCODE Data Links (Direct)

| Cell Line | DNase | CTCF | H3K27ac | H3K4me3 | H3K4me1 | Status |
|-----------|-------|------|---------|---------|---------|--------|
| GM12878 | ENCFF | ENCFF | ENCFF | ENCFF | ENCFF | Ready |
| K562 | ENCFF | ENCFF | ENCFF | ENCFF | ENCFF | Ready |
| HeLa-S3 | ENCFF | ENCFF | ENCFF | ENCFF | ENCFF | Ready |
| HUVEC | ENCFF774IVP | ENCFF | ENCFF | ENCFF | ENCFF | To Add |
| HepG2 | ENCFF | ENCFF | ENCFF | ENCFF | ENCFF | To Add |
| H1-hESC | ENCFF | ENCFF | ENCFF | ENCFF | ENCFF | To Add |

### Estimated Processing Time (RTX 5070 Ti)

| Task | Time per Cell Line |
|------|-------------------|
| Download bigWig files | ~30 min |
| Process bigWig → .pt | ~15 min |
| Process narrowPeak → .pt | ~2 min |
| Total per cell line | ~50 min |

---

## Troubleshooting

### Common Issues

1. **Missing histone marks**: Some cell lines lack certain marks. Use zeros for missing features.

2. **File naming mismatch**: Ensure file names match the JSON config exactly.

3. **Memory errors during processing**: Process one cell line at a time.

4. **CUDA out of memory**: Reduce batch_size from 64 to 32.

### Validation Script

```python
# Verify all cell lines are properly configured
import json
import os

with open('data/genomic_data/CTCF_DNase_6histone_local.500.json') as f:
    config = json.load(f)

base_path = config.pop('_location')
for cell, marks in config.items():
    print(f"\n{cell}:")
    for mark, filename in marks.items():
        full_path = os.path.join(base_path, filename)
        exists = os.path.exists(full_path)
        print(f"  {mark}: {'OK' if exists else 'MISSING'} - {filename}")
```

---

## References

1. **BENGI**: Moore et al. (2020) "A curated benchmark of enhancer-gene interactions for evaluating enhancer-target gene prediction methods" *Genome Biology*
2. **ENCODE**: ENCODE Project Consortium (2012) "An integrated encyclopedia of DNA elements in the human genome" *Nature*
3. **KansformerEPI**: Original architecture paper for EPI prediction with KAN
4. **POCD-ND**: Position-aware Oligonucleotide Composition Density encoding

---

*Last updated: March 2026*
