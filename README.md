# Identification of a type 1 diabetes-associated T cell receptor repertoire signature from the human peripheral blood

## 📄Abstract

Type 1 Diabetes (T1D) is a T-cell mediated disease with a strong immunogenetic HLA dependence. HLA allelic influence on the T cell receptor (TCR) repertoire shapes thymic selection and controls activation of diabetogenic clones, yet remains largely unresolved in T1D. We sequenced the circulating TCRβ chain repertoire from 2250 HLA-typed individuals across three cross-sectional cohorts, including T1D patients, and healthy related and unrelated controls. We found that HLA risk alleles show higher restriction of TCR repertoires in T1D individuals. Machine learning analysis yielded AUROC of 0.77 on test cohorts for T1D classification. T1D-specific TCR features predominantly localized to the subsequence motifs, indicating absence of T1D-associated public clones. These TCR motifs were also observed in independent TCR cohorts residing in pancreas-draining lymph nodes of T1D individuals. Collectively, our data demonstrate T1D-related TCR motif enrichment based on genetic risk, offering a potential metric for autoreactivity and basis for TCR-based diagnostics and therapeutics.

---

## 📥 Data and Code Availability 

- Additional **Code** and **datasets** will be made available in response to community feedback and demand.
- **Paper Reference**:  
  *[Identification of a type 1 diabetes-associated T cell receptor repertoire signature from the human peripheral blood](https://www.medrxiv.org/content/10.1101/2024.12.10.24318751v1)*
- The repertoire data will be uploaded in **[immuneACCESS database](https://clients.adaptivebiotech.com/)** after publication.
- We will update the DOI here shortly.

📝 Materials in this repository are licensed under **CC-BY 4.0 International**.  

---
## 🔧 Tools and Scripts  

### **Repertoire-Level Similarity and Diversity**  
- **Percentage of shared public clones** & **Morisita-Horn (MH) index**  
  - Tool: [CompAIRR](https://github.com/uio-bmi/compairr)  
  - Analysis outputs are provided in the **data folder**.

### **Disease-Associated CDR3β Sequences**  
- Databases:  
  - [McPAS-TCR](https://friedmanlab.weizmann.ac.il/McPAS-TCR/)  
  - [VDJdb](https://vdjdb.cdr3.net/)  

### **HLA-Associated Public TCRs (TCR Clustering)**  
- Tool: [tcrdist3](https://github.com/phbradley/tcr-dist)  
- Results are included in the **data section**.

### **HLA Risk Alleles based TCR repertoire restriction**  
- Resource: [cdr3-QTL](https://github.com/immunogenomics/cdr3-QTL/tree/main)  
- Additional results and codes are provided in this repository.

### **Statistical Classification Using Public Clones**  
- Tool: [ImmuneML](https://github.com/uio-bmi/immuneML)  
- YAML configuration files are included in the **code section**.

### **K-mer Based Logistic Regression Classification**  
- Complete code for **K-mer classification** is provided here.

### **Deep Learning Classification with DeepRC**  
- Tool: [DeepRC](https://github.com/ml-jku/DeepRC)  

### **Quantitative Trait Locus (QTL) Analysis**  
- Outputs of the QTL analysis are included in the **data folder**.

---

## 🛠️ How to Cite  
If you use this work, please cite:  
> *"Identification of a type 1 diabetes-associated T cell receptor repertoire signature from the human peripheral blood."*  
> [DOI Link](https://www.medrxiv.org/content/10.1101/2024.12.10.24318751v1)  

---

## 💡 Contributing  
We welcome contributions to improve the code and analysis.
If you'd like us to add or update any information in this repository, please **contact us directly** using the mail below.  

---

## 👌 Acknowledgments  
We thank the incredible collaborators and funding agencies that made this work possible.  

🚀 **Get in Touch**  
For inquiries or collaboration, please feel free to open an issue or reach out via puneet.rawat[at]medisin.uio.no.

---

### 🌟 **Let's decode T1D together!**  

