# AI Use Disclosure

**Tool:** Claude/Gemini

## Log

**1. Understanding the assignment and planning**
- Question: "How should I start this project, explain in detail" and "make a detailed roadmap".
- Claude read the assignment document and gave a phased plan (repository setup, EDA, splitting, preprocessing, model runs, test evaluation, K-Means, Wine, CSV files, report, video) and common pitfalls (data leakage, `ddof=1`, macro averaging, ROC-AUC with malignant as the positive class).
- How I used it: as a checklist and time plan. I wrote the code and text myself.

**2. Code written by AI**
- Question: I asked how the code should look, for helper functions, and later for notebook cells.
- Claude/Gemini wrote the first versions of: stratified splitting and split checks, scaling per seed, the model grid, `compute_metrics`, `run_one_model`, `run_one_seed`, the confusion matrix and the manual metric check, the ROC plots, the K-Means code, the cluster plot, and the CSV save/combine functions.
- How I used it: I integrated the generated code into my project, moved some key functions into notebook cells for better explanation during the presentation, and wrote all markdown documentation myself after verifying the logic.

**3. Environment and tooling setup**
- Questions: creating notebooks and Markdown headings in VS Code, which packages to list in `requirements.txt`, creating a virtual environment.
- Claude gave VS Code steps and PowerShell commands (`python -m venv`, activation, `pip install -r requirements.txt`). Its first package list copied versions from my `pip freeze` and failed because numpy 2.5.x needs Python 3.12 or later; I use Python 3.11.9.
- How I used it: I installed the packages without pinned versions and wrote `requirements.txt` from the versions actually installed in my virtual environment.

**4. Course slides**
- Question: whether the instructor's slides contain example code for the assignment steps.
- Claude read the slide decks (validation methods, pre-processing, decision trees, KNN, Bayes, K-Means) and listed where code exists and where it differs from the assignment (for example weighted instead of macro averaging, scaling the full dataset before splitting, no stratification).
- How I used it: to know which slides to look.

**5. Concept explanations**
- Questions: how `predict_proba` columns relate to `classes_`, why the test set uses the seed-42 scaler, what `**` dictionary unpacking and `groupby().agg(["mean", "std"])` with multi-level column names do, how multi-key sorting works, why Jupyter shows only the last expression of a cell, what the 22-row summary table represents, and why `compute_metrics` sits in Section 7.
- Gemini explained the concepts in words.
- How I used it: to understand my code so that I can explain it in the video and in class.

**6. Errors and missing parts**
- `NameError` for `result`: the cell that defines it was run after the cell that uses it. Fix: run the cells in order.
- The 9-row split table and the 22-row summary table were not shown because Jupyter only displays the last expression of a cell. Fix: separate cells.
- The ROC plot was called a second time in 9.4. Fix: removed.
- `manual_check` printed TP, FP and FN only. Fix: added TN.
- Sections 7 and 8.2 repeated the same model runs. Fix: the runs stay in Section 7 and Section 8.2 only does the summary and the selection.
- Missing parts that I added with Claude's code: per-column missing counts, the class-label mapping, the class distribution and correlation figures, the package versions, the Section 6 row-index check, and the CSV export cell.

**7. Review of my code**
- I shared my `helpers.py` and notebooks several times and asked whether anything was wrong.
- Claude pointed out problems: tables not displayed because they were not the last expression of a cell, a duplicated ROC call, out-of-order execution counts, a truncated 66-row table, an unused variable in Section 6, a comment that did not match the code, and missing section interpretations. It also checked my output against its own independent recomputation (split sizes, selected configuration, test metrics, K-Means results) in its own environment.
- How I used it: I made the corrections.
- - Gemini pointed out wrong statements in my Markdown (for example Q1/Q3 swapped in the boxplot description, a median reported as a mean, the wrong validation score for one configuration, the K-Means ARI described as a percentage, silhouette direction), and missing interpretations.
- How I used it: I corrected my text. 

**8. Scaler comparison**
- Question: why MinMaxScaler gave a slightly higher validation score than StandardScaler.
- Claude explained that the difference equals one validation sample out of 171 and that choosing a scaler after seeing results conflicts with assignment 5.2.
- How I used it: I kept StandardScaler, which I had chosen before running the experiments.

