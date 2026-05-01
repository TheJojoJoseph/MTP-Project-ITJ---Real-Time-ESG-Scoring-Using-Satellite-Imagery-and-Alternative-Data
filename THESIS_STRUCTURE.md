# M.Tech Thesis Structure Guide

**Thesis Title:** Real-Time ESG Scoring Using Satellite Imagery and Alternative Data  
**Student:** Jojo Joseph (M24DE3041)  
**Supervisor:** Dr. Puneet Sharma  
**File:** `ReportProject/MTP-template-IITJ/main.pdf` (42 pages, 1.04 MB)

---

## 📑 Thesis Organization

### Front Matter (Roman numerals: i-viii)

| Section | File | Page | Description |
|---------|------|------|-------------|
| **Title Page** | `titlePage.tex` | i | Thesis title, author, institution, date |
| **Declaration** | `declaration.tex` | ii | Originality statement |
| **Certificate** | `certificate.tex` | iii | Supervisor certification |
| **Acknowledgments** | `acknowledgments.tex` | iv | Thanks and credits |
| **Abstract** | `abstract.tex` | v | 4-paragraph comprehensive summary |
| **Table of Contents** | Auto-generated | vi | Section/subsection listing |
| **List of Figures** | Auto-generated | vii | 7 figures with captions |
| **List of Tables** | Auto-generated | viii | 10 tables with captions |
| **List of Abbreviations** | `abbreviations.tex` | ix | 24 technical terms (NEW) |

### Main Content (Arabic numerals: 1-42)

| Section | Pages | Key Content |
|---------|-------|-------------|
| **1. Introduction** | 1-4 | Motivation, alternative data, deep learning, objectives |
| **2. Literature Survey** | 5-8 | ESG ratings, satellite RS, ViT, GNN, streaming, multi-modal |
| **3. Problem Definition** | 9-11 | Formal problem statement, challenges, objectives, scope |
| **4. Methodology** | 12-22 | Architecture, data ingestion, ViT, GNN, scoring, streaming |
| **5. Experimental Findings** | 23-35 | Setup, results, ablation, statistics, streaming, visualization |
| **6. Summary & Future Work** | 36-38 | Contributions, implications, future directions |
| **References** | 39-42 | 50 references (IEEEtran style) |

---

## 🔬 Section 4: Methodology Breakdown

### 4.1 System Architecture (p. 12)
- 5-layer pipeline overview
- Figure 1: Architecture diagram

### 4.2 Data Ingestion (p. 13-14)
- Synthetic company universe (50 companies, 5 sectors)
- Satellite imagery simulation (224×224 RGB)
- Corporate registry data (OpenCorporates-style)

### 4.3 Vision Transformer (p. 15-16)
- Model: `google/vit-base-patch16-224`
- **Algorithm 1:** ViT Feature Extraction (NEW)
- Time complexity: O(P² d)
- Output: 768-dim embeddings

### 4.4 Graph Neural Networks (p. 17-20)
- Graph construction (390 edges, density 0.318)
- **Algorithm 2:** GraphSAGE Message Passing (NEW)
- **Algorithm 3:** Graph Attention Network (NEW)
- Time complexity: O(K |E| d)

### 4.5 ESG Scoring Engine (p. 21-22)
- **Algorithm 4:** ESG Score Computation (NEW)
- Transparent formulas: E, S, G subscores
- Weighted aggregation: ESG = 0.35E + 0.35S + 0.30G

### 4.6 Real-Time Streaming (p. 23-24)
- **Algorithm 5:** Streaming Updates (NEW)
- Kafka-like event processing
- Latency: 20-50ms per event

---

## 📊 Section 5: Experimental Findings Breakdown

### 5.1 Experimental Setup (p. 23-24)
- **Table 1:** Comprehensive configuration (30+ parameters) (NEW)
- Hardware: Apple M1 GPU
- Dataset: 50 companies, 5 sectors, 3 regions

### 5.2 Static ESG Distribution (p. 25-26)
- **Table 2:** Top 10 ESG performers
- Mean ESG: 64.07 ± 5.62
- Component breakdown: E=64.4, S=60.6, G=67.8

### 5.3 Ablation Study (p. 27-28) ⭐ NEW
- **Table 3:** 5 model variants with statistical tests
- Full model vs. baselines: all p < 0.05
- ViT contribution: Δ = -2.62, p = 0.012
- GNN contribution: Δ = -1.76, p = 0.045

### 5.4 Statistical Analysis (p. 29-30) ⭐ NEW
- **ANOVA:** F(4,45) = 3.87, p = 0.008
- **Tukey HSD:** Pairwise sector comparisons
- **Table 4:** 95% Confidence intervals by sector
- Manufacturing vs. Energy: Δ = 5.80, p = 0.004

### 5.5 Graph Neural Network Impact (p. 31)
- **Table 5:** Graph statistics
- Density = 0.318 (optimal balance)
- GNN smoothing reduces variance by ~15%

### 5.6 Streaming Updates (p. 32-33)
- **Figure 6:** Real-time score evolution
- Update latency: 20-50ms
- Score volatility: 1.5-3.2 across sectors

### 5.7 Visualization (p. 34-35)
- **Figure 7:** ViT attention heatmaps
- **Figure 4:** Company graph visualization
- Interpretability analysis

### 5.8 Computational Performance (p. 36)
- **Table 6:** Runtime breakdown
- ViT: 12.3 sec (50 companies)
- GNN: 0.6 sec
- Total pipeline: ~15 seconds

---

## 🎯 Key Algorithms (NEW Content)

All 5 algorithms include:
- **Input/Output:** Formal specification
- **Step-by-step pseudocode:** Using algorithm2e package
- **Time Complexity:** Big-O analysis
- **Space Complexity:** Memory requirements

### Algorithm 1: ViT Feature Extraction (p. 16)
```
Input: Satellite image I_i, Pre-trained ViT model
Output: 768-dim embedding h_i^ViT
Steps: Resize → Normalize → Patch → Attention (12 layers) → Extract [CLS]
Complexity: O(P² d) time, O(P·d + L·d²) space
```

### Algorithm 2: GraphSAGE (p. 20)
```
Input: Company graph G, Node features {f_i}
Output: Company embeddings {z_i}
Steps: Layer 1 aggregation → Layer 2 refinement → Normalize
Complexity: O(|E|·d + N·d²) time, O(N·d + |E|) space
```

### Algorithm 3: GAT (p. 21)
```
Input: Graph G, K attention heads, Weight matrices
Output: Multi-head embeddings {z_i}
Steps: Compute attention logits → Normalize → Aggregate → Concatenate
Complexity: O(K·|E|·d) time, O(N·K·d + |E|·K) space
```

### Algorithm 4: ESG Scoring (p. 22)
```
Input: ViT embedding, GNN embedding, Tabular features
Output: ESG score and E/S/G subscores
Steps: Extract components → Compute E/S/G → Weighted aggregation
Complexity: O(d) time, O(1) space
```

### Algorithm 5: Streaming (p. 24)
```
Input: Event stream S, Current ESG scores
Output: Updated scores over time
Steps: Poll event → Update features → Recompute affected scores → Emit
Latency: 20-50ms per event
```

---

## 📈 Key Tables (NEW Content)

### Table 1: Experimental Setup (p. 24)
- Dataset configuration (50 companies, 5 sectors)
- ViT configuration (768-dim, 12 layers, 12 heads)
- GNN configuration (GraphSAGE, 2 layers, 256→128 dim)
- ESG scoring weights (0.35, 0.35, 0.30)
- Hardware & software versions

### Table 2: Top 10 Performers (p. 25)
- Company ID, sector, E/S/G subscores, overall ESG
- Manufacturing dominance (3 of top 5)
- Highest: C014 (Manufacturing) = 73.06

### Table 3: Ablation Study (p. 27) ⭐
- 5 model variants: Full, Baseline, ViT-only, GNN-only, Random
- Mean ESG, Std Dev, Δ vs Full, p-value
- All components significant (p < 0.05)

### Table 4: 95% Confidence Intervals (p. 30) ⭐
- Sector-level mean ESG ± 95% CI
- Manufacturing: 65.96 ± 2.21 (narrowest)
- Energy: 60.16 ± 5.08 (widest)

### Table 5: Graph Statistics (p. 31)
- 50 nodes, 390 edges, density 0.318
- Average degree 15.6
- GNN impact on score variance

### Table 6: Computational Performance (p. 36)
- Component-wise runtime breakdown
- ViT: 12.3 sec (50 images on M1 GPU)
- Total: ~15 seconds end-to-end

---

## 🎨 Key Figures

### Figure 1: System Architecture (p. 12)
- 5-layer pipeline: Ingestion → ViT → GNN → Scoring → Streaming
- Data flow diagram

### Figure 2: Score Distributions (p. 26)
- Histograms for E, S, G, and overall ESG scores
- Normal-like distributions with sector patterns

### Figure 3: Sector Comparison (p. 28)
- Boxplots by sector
- Manufacturing > Technology > Finance > Healthcare > Energy

### Figure 4: Company Graph (p. 34)
- Network visualization (nodes = companies, edges = relationships)
- Node size ∝ ESG score
- Color = sector

### Figure 5: Correlation Heatmap (p. 32)
- Feature correlation matrix
- Strong correlations: emissions↔E (-0.71), board_indep↔G (0.68)

### Figure 6: Streaming Scores (p. 33)
- Time series of 3 companies over 100 seconds
- Different volatility patterns by sector

### Figure 7: Attention Heatmap (p. 35)
- ViT attention visualization on satellite images
- Energy vs. Finance comparison

---

## 📚 Bibliography (50 References)

### Breakdown by Topic
- **ESG Finance & Ratings:** 8 papers (Friede 2015, Berg 2019, etc.)
- **Satellite Remote Sensing:** 6 papers (Gorelick 2017, Hansen 2013, etc.)
- **Vision Transformers:** 8 papers (Dosovitskiy 2021, Bazi 2021, etc.)
- **Graph Neural Networks:** 7 papers (Hamilton 2017, Veličković 2018, etc.)
- **Real-Time Streaming:** 5 papers (Kreps 2011, Carbone 2015, etc.)
- **Multi-Modal Learning:** 4 papers (Baltrusaitis 2018, Jean 2016, etc.)
- **Recent (2024-2026):** 12 papers ⭐ (Zhang 2024, Kumar 2024, etc.)

### Citation Style
- **Format:** IEEEtran (numbered references)
- **In-text:** \cite{friede2015esg} → [1]
- **Bibliography:** Alphabetical by first author

---

## 🔧 LaTeX Compilation

### Prerequisites
```bash
# Install Tectonic (modern LaTeX compiler)
brew install tectonic  # macOS
# or use your package manager
```

### Compile Thesis
```bash
cd ReportProject/MTP-template-IITJ
tectonic main.tex
# Output: main.pdf (42 pages, 1.04 MB)
```

### File Dependencies
```
main.tex includes:
├── titlePage.tex
├── declaration.tex
├── certificate.tex
├── acknowledgments.tex
├── abstract.tex
├── abbreviations.tex (NEW)
└── mainReport.tex (main content with algorithms)

Bibliography:
└── refs.bib (50 entries, IEEEtran.bst style)
```

### Packages Used
- **algorithm2e:** Algorithm blocks with line numbers
- **amsmath, amssymb:** Mathematical symbols
- **graphicx:** Figure inclusion
- **hyperref:** Cross-references and links
- **IEEEtran:** Bibliography style
- **datetime:** Custom date formatting

---

## 📝 Writing Standards

### IIT Thesis Requirements Met
✅ Front matter (title, declaration, certificate, acknowledgments, abstract)  
✅ Table of contents, list of figures, list of tables  
✅ List of abbreviations (NEW)  
✅ Main sections with proper numbering  
✅ References in standard format (IEEEtran)  
✅ Formal problem definition  
✅ Algorithm specifications with complexity (NEW)  
✅ Experimental validation with statistics (NEW)  
✅ Ablation studies (NEW)  
✅ Confidence intervals (NEW)  
✅ Professional formatting (12pt, 1.5 spacing)  

### Academic Rigor
- **Hypothesis Testing:** ANOVA, t-tests, p-values < 0.05
- **Effect Sizes:** Mean differences with confidence intervals
- **Reproducibility:** Seed=42, all hyperparameters documented
- **Transparency:** Synthetic data limitations acknowledged
- **Validation:** Ablation study confirms component contributions

---

## 🎓 Submission Checklist

Before final submission, verify:

- [ ] Supervisor name correct: Dr. Puneet Sharma ✅
- [ ] Roll number correct: M24DE3041 ✅
- [ ] Date correct: May 2026 ✅
- [ ] All figures numbered and captioned ✅
- [ ] All tables numbered and captioned ✅
- [ ] All citations in bibliography ✅
- [ ] No broken cross-references ✅
- [ ] Abstract under 500 words ✅
- [ ] Acknowledgments professional ✅
- [ ] PDF file size reasonable (1.04 MB) ✅
- [ ] Page count appropriate (42 pages) ✅
- [ ] No typos or grammatical errors
- [ ] Consistent notation throughout
- [ ] Algorithm numbering sequential ✅
- [ ] Equation numbering sequential ✅

---

## 📞 Quick Reference

**Main PDF:** `ReportProject/MTP-template-IITJ/main.pdf`  
**LaTeX Source:** `ReportProject/MTP-template-IITJ/main.tex`  
**Bibliography:** `ReportProject/MTP-template-IITJ/refs.bib`  
**Figures:** `ReportProject/MTP-template-IITJ/Figures/`  

**Compile Command:** `tectonic main.tex`  
**View PDF:** `open main.pdf` (macOS) or `xdg-open main.pdf` (Linux)  

**Repository:** https://github.com/TheJojoJoseph/MTP-2-IITJ  
**Commit:** `09d5d91` (Latest with comprehensive README)

---

**Document Version:** 1.0  
**Last Updated:** May 2, 2026  
**Status:** ✅ Ready for Submission
