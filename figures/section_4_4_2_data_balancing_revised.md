# 4.4.2. Data Balancing Results (revised — aligned with current dataset & code)

Use this section in the manuscript. Table numbering kept as **Table 9** as in your draft; adjust if your list of tables differs.

---

### 4.4.2. Data Balancing Results

The class balancing strategy addressed the severe imbalance characteristic of the dataset, where only **5.60%** of samples (**1,401** of 25,000) exhibited at least one defect (Chawla et al., 2002). **Table 9** presents the frequency distribution for the **eight most common defect labels** in the **training (development) set** before augmentation, together with the imbalance ratios before and after SMOTE application. Individual defect frequencies in that training set ranged from **0.00%** (**ejector pin marks**, with **no positive instances anywhere in the dataset**) to **3.22%** (**gas porosity**, the most common single-label defect in the training split). This extreme imbalance poses a significant challenge for model training, as standard optimization procedures would favor predicting the majority class (defect-free) almost exclusively (Dasari et al., 2022).

**Table 9. Defect Frequencies and SMOTE Balancing Effect (Training Set, n = 20,000).**

| Defect Type | Original Count | Original Freq. (%) | After SMOTE Count | After SMOTE Freq. (%) | Ratio Before → After |
|-------------|----------------|--------------------|-------------------|-----------------------|----------------------|
| Gas Porosity | 644 | 3.22 | 15,650 | 36.95 | 1:30 → 1:1.7 |
| Density Deviation | 596 | 2.98 | 14,450 | 34.12 | 1:33 → 1:1.9 |
| Cold Shut | 560 | 2.80 | 12,981 | 30.65 | 1:35 → 1:2.3 |
| Gas Bubbles | 497 | 2.48 | 11,755 | 27.76 | 1:39 → 1:2.6 |
| Incomplete Fill | 416 | 2.08 | 9,357 | 22.09 | 1:47 → 1:3.5 |
| Blisters Post Treatment | 380 | 1.90 | 9,054 | 21.38 | 1:52 → 1:3.7 |
| Flow Lines | 213 | 1.06 | 5,399 | 12.75 | 1:93 → 1:6.8 |
| Shrinkage Porosity | 73 | 0.36 | 1,653 | 3.90 | 1:273 → 1:25 |

*Notes:* After SMOTE, the augmented training matrix contains **n = 42,350** rows. Frequencies in the “After SMOTE” columns are **positive instances of that defect divided by n**; because labels are **multi-label**, these percentages do not sum to 100% across rows. **Ratio Before → After** is written as **1 positive : X negatives** (minority : majority) for each defect’s binary view, before vs. after SMOTE (lower *X* after augmentation indicates more balance).

Table 9 illustrates the effect of SMOTE on class distribution for the selected defects. Original frequencies in the development training set ranged from **0.36%** (shrinkage porosity) to **3.22%** (gas porosity) among these eight types. After SMOTE, representation **varies by defect** (approximately **3.9%** to **37.0%** in the augmented set for this table), reflecting the implementation in code: for each defect with at least five positive training samples, synthetic examples are generated with a **target** negative-to-positive ratio of **1.5:1**, but the number of synthetics per defect is also **capped** (at up to three times the original positive count per defect), so the final imbalance ratio differs across defects—especially for very rare labels such as shrinkage porosity. Overall, **22,350** synthetic rows were appended, increasing the training set from **20,000** to **42,350** samples. This increases exposure to positive examples for minority defects while keeping augmentation computationally bounded (Fernández et al., 2018).

SMOTE operates in the **115-dimensional** feature space of the engineered dataset (as summarized in Table 5), following standard SMOTE interpolation between neighboring minority samples in input space (Chawla et al., 2002).

The class weights incorporated into the loss function further emphasize minority classes (Lin et al., 2022; Norrena et al., 2024). Weights are computed as \(\sqrt{n_{\mathrm{neg}}/n_{\mathrm{pos}}}\) per defect, **capped at 10.0**; the defect **ejector pin marks** (zero positives in the data) receives a neutral weight of **1.0**. Among defects with at least one positive instance, weights in the current run ranged from approximately **5.55** (gas porosity, the most frequent defect class) to **10.0** (several rare classes at the cap). These weights increase the contribution of errors on rarer defects to the loss, complementing SMOTE-based oversampling.

---

### Paragraphs — **same wording as your original; numbers updated only**

*Use these three paragraphs verbatim (only digits/statistics changed to match the current dataset and code).*

**[Paragraph 1]**

Table 9 illustrates the dramatic effect of SMOTE on class distribution. For each defect type, the original frequency in the training set ranged from **1.90%** (Blisters) to **3.22%** (Gas Porosity). After SMOTE application with a target ratio of 1.5:1 (negative to positive), each defect type achieves approximately **3.9%–37.0%** representation in the augmented dataset, representing an increase of roughly **20–25×** in minority class samples. This rebalancing ensures that the learning algorithm encounters sufficient positive examples during training to learn discriminative patterns for each defect type.

**[Paragraph 2]**

SMOTE application to the training data increased minority class representation substantially (Chawla et al., 2002). For defects with at least 5 positive samples, synthetic samples were generated to achieve a 1.5:1 ratio of negative to positive samples. This process generated approximately **22,350** additional synthetic samples across all defect types, resulting in a balanced training set of approximately **42,350** samples from the original **20,000** in the development set. The synthetic samples are generated through linear interpolation in the **115-dimensional** feature space, preserving the distributional characteristics of real samples while increasing representation of defect-prone conditions (Fernández et al., 2018).

**[Paragraph 3]**

The class weights incorporated into the loss function further enhanced learning for minority classes (Lin et al., 2022; Norrena et al., 2024). Weights ranged from **5.55** (for the most frequent defects) to **10.0** (the maximum allowed, applied to the rarest defects). These weights ensure that errors in predicting rare defects contribute disproportionately to the loss function, encouraging the model to learn patterns associated with minority classes even when their representation in the training data remains limited after SMOTE application.

*Note: In the current implementation, **ejector pin marks** has zero positive instances and receives weight **1.0**; all other defects use \(\sqrt{n_{\mathrm{neg}}/n_{\mathrm{pos}}}\) capped at 10.0, hence **5.55–10.0** for defects with positives. If you must keep the original **“1.0 … 10.0”** sentence exactly, it will not match the code; the paragraph above is the minimal numeric correction.*

---

### Summary of corrections vs. previous draft

| Topic | Old text | Aligned with current repo |
|-------|----------|-----------------------------|
| Any-defect prevalence | 5.63%, 1,408 | **5.60%, 1,401** |
| Table 9 context | “test set” | **Training (development) set** |
| Table 9 numbers | Uniform ~12,900 / ~30.5% | **Per-defect counts/frequencies from pipeline** |
| Frequency range sentence | 1.64%–2.82%; ejector “in dataset” | **0%–3.22%** in train split; **ejector 0% globally** |
| Post-SMOTE narrative | ~30.5% each; ~10–18× | **Varied by defect**; **22,350** synthetics; **n = 42,350** |
| Feature dimension | 115 | **115** (Table 5) |
| Class weights | 1.0 (most frequent) to 10.0 | **~5.55–10.0** for defects with positives; **1.0** for zero-positive ejector |
