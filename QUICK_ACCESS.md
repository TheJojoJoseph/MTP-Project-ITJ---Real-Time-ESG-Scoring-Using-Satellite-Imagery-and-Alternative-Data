# 🚀 Quick Access Guide - M.Tech Thesis

**Repository:** https://github.com/TheJojoJoseph/MTP-2-IITJ

---

## 📄 Download Thesis PDF

**Direct Link:** [main.pdf](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/ReportProject/MTP-template-IITJ/main.pdf)

**Or via command line:**
```bash
git clone https://github.com/TheJojoJoseph/MTP-2-IITJ.git
cd MTP-2-IITJ
open ReportProject/MTP-template-IITJ/main.pdf
```

**File Info:**
- **Pages:** 42
- **Size:** 1.04 MB
- **Supervisor:** Dr. Puneet Sharma
- **Date:** May 2026

---

## 📊 View Experimental Results

### Figures (7 PDFs)
```bash
ls experiments/results/figures/
```

1. **architecture.pdf** - System pipeline diagram
2. **company_graph.pdf** - Network visualization (50 companies)
3. **score_distributions.pdf** - E/S/G/ESG histograms
4. **sector_comparison.pdf** - Boxplots by sector
5. **correlation_heatmap.pdf** - Feature correlation matrix
6. **streaming_scores.pdf** - Real-time score evolution
7. **attention_heatmap.pdf** - ViT attention patterns

### Data (5 CSV files)
```bash
ls experiments/results/data/
```

1. **esg_scores.csv** - Final ESG scores for 50 companies
2. **companies_raw.csv** - Company features
3. **top_10_performers.csv** - Best ESG performers
4. **bottom_10_performers.csv** - Lowest ESG scores
5. **sector_analysis.csv** - Sector-level statistics

---

## 📚 Documentation Files

| File | Description | Link |
|------|-------------|------|
| **README.md** | Complete thesis overview | [View](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/README.md) |
| **THESIS_STRUCTURE.md** | Section-by-section breakdown | [View](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/THESIS_STRUCTURE.md) |
| **EXPERIMENTAL_FINDINGS.md** | Detailed results (15 pages) | [View](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/EXPERIMENTAL_FINDINGS.md) |
| **PROJECT_COMPLETION_SUMMARY.md** | Project overview | [View](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/PROJECT_COMPLETION_SUMMARY.md) |

---

## 🎯 Key Thesis Sections (Page Numbers)

### Algorithms (5 total, all with complexity analysis)
- **Algorithm 1:** ViT Feature Extraction - Page 16
- **Algorithm 2:** GraphSAGE Message Passing - Page 20
- **Algorithm 3:** Graph Attention Network - Page 21
- **Algorithm 4:** ESG Score Computation - Page 22
- **Algorithm 5:** Real-Time Streaming - Page 24

### Statistical Analysis
- **ANOVA Test:** Page 29 (F=3.87, p=0.008)
- **Ablation Study:** Page 27-28 (Table 3)
- **95% Confidence Intervals:** Page 30 (Table 4)
- **Tukey HSD:** Page 29 (pairwise comparisons)

### Key Results Tables
- **Table 1:** Experimental Setup - Page 24
- **Table 2:** Top 10 Performers - Page 25
- **Table 3:** Ablation Study - Page 27 ⭐
- **Table 4:** 95% CI by Sector - Page 30 ⭐
- **Table 5:** Graph Statistics - Page 31
- **Table 6:** Performance Breakdown - Page 36

---

## 🔬 Run Experiments Yourself

### 1. Clone and Install
```bash
git clone https://github.com/TheJojoJoseph/MTP-2-IITJ.git
cd MTP-2-IITJ
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python experiments/standalone_esg_experiment.py
```

**Outputs:**
- ✅ All 7 figures in `experiments/results/figures/`
- ✅ All 5 CSV datasets in `experiments/results/data/`
- ✅ Summary JSON in `experiments/results/summary.json`
- ✅ Runtime: ~5 minutes on M1 GPU

### 3. View Interactive Notebook
```bash
jupyter notebook notebooks/esg_scoring_pipeline.ipynb
```

---

## 📖 Bibliography Highlights

**Total References:** 50

### Recent Papers (2024-2026) - 20 references
- Zhang et al. 2024: Deep learning for ESG rating prediction
- Kumar et al. 2024: Satellite imagery and transformers
- Liu et al. 2024: GNN for financial risk
- Brown et al. 2024: AI for climate finance
- Wang et al. 2025: Real-time streaming GNN

### Foundational Works
- Dosovitskiy et al. 2021: Vision Transformers (ViT)
- Hamilton et al. 2017: GraphSAGE
- Veličković et al. 2018: Graph Attention Networks (GAT)
- Friede et al. 2015: ESG and financial performance
- Berg et al. 2019: ESG rating divergence

---

## 💻 Compile LaTeX Source

### Requirements
```bash
# macOS
brew install tectonic

# Or download from: https://tectonic-typesetting.github.io
```

### Compile
```bash
cd ReportProject/MTP-template-IITJ
tectonic main.tex
```

**Output:** `main.pdf` (42 pages)

### Edit and Recompile
```bash
# Edit any .tex file
vim mainReport.tex

# Recompile
tectonic main.tex
```

---

## 📈 Key Findings Summary

### Overall Performance
- **Mean ESG Score:** 64.07 ± 5.62
- **Range:** [51.46, 73.1]
- **Components:** E: 64.4, S: 60.6, G: 67.8
- **Best Performer:** C014 (Manufacturing) - 73.06

### Sector Rankings (with 95% CI)
1. **Manufacturing:** 65.96 ± 2.21
2. **Technology:** 64.74 ± 3.38
3. **Finance:** 63.81 ± 4.22
4. **Healthcare:** 63.14 ± 4.34
5. **Energy:** 60.16 ± 5.08

### Statistical Validation
- **ANOVA:** F(4,45) = 3.87, p = 0.008 ✅
- **Manufacturing vs Energy:** Δ = 5.80, p = 0.004 ✅
- **All model components:** p < 0.05 ✅

### Ablation Study
- **ViT contribution:** +2.62 points (p = 0.012)
- **GNN contribution:** +1.76 points (p = 0.045)
- **Full model best:** 64.07 (baseline: 58.23)

---

## 🎓 Citation

```bibtex
@mastersthesis{joseph2026realtime,
  title={Real-Time ESG Scoring Using Satellite Imagery and Alternative Data},
  author={Joseph, Jojo},
  year={2026},
  school={Indian Institute of Technology Jodhpur},
  department={Data Science and Engineering},
  type={M.Tech Thesis},
  supervisor={Dr. Puneet Sharma}
}
```

---

## 🔗 Important Links

- **Repository:** https://github.com/TheJojoJoseph/MTP-2-IITJ
- **Thesis PDF:** [main.pdf](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/ReportProject/MTP-template-IITJ/main.pdf)
- **Documentation:** [README.md](https://github.com/TheJojoJoseph/MTP-2-IITJ/blob/main/README.md)
- **Experiments:** `experiments/standalone_esg_experiment.py`
- **Results:** `experiments/results/`

---

## ✅ Submission Checklist

### Required Files
- [x] **main.pdf** - Final thesis (42 pages)
- [x] **thesis.zip** - Complete LaTeX source
- [x] **Source code** - `esg_pipeline/` modules
- [x] **Experimental results** - `experiments/results/`
- [x] **Bibliography** - 50 references in `refs.bib`
- [x] **Documentation** - README, structure guide, findings

### Quality Checks
- [x] Supervisor: Dr. Puneet Sharma ✅
- [x] Roll number: M24DE3041 ✅
- [x] Algorithms: 5 with complexity ✅
- [x] Statistical tests: ANOVA, t-tests ✅
- [x] Confidence intervals: 95% CI ✅
- [x] Ablation study: 5 variants ✅
- [x] Bibliography: 50 refs ✅
- [x] Page count: 42 pages ✅

---

## 📞 Contact

**Student:** Jojo Joseph  
**Roll Number:** M24DE3041  
**Email:** Available in thesis  
**Institution:** IIT Jodhpur  
**Department:** Data Science and Engineering  

---

**Last Updated:** May 2, 2026  
**Status:** ✅ Ready for Submission  
**Commits:** 3 (Latest: 0efe41b)
