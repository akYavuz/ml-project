# Classification and Clustering on the Breast Cancer and Wine Datasets

## 1. Installing dependencies

Use Python 3.11.9. Then install the packages from `requirements.txt`:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Running the project

1. Open `notebooks/breast_cancer.ipynb` and run all cells.
2. Open `notebooks/wine.ipynb` and run all cells.

`helpers.py` must stay one folder above the notebooks, because they import it with `sys.path.append("..")`. Figures are saved to `figures/` and CSV results to `results/`. The combined CSV files in `results/` are created after both notebooks have been run (the first one prints "waiting for the other notebook", which is normal).

## 3. Python and package versions

| Package | Version |
|---|---|
| Python | 3.11.9 |
| ipykernel | 7.4.0 |
| matplotlib | 3.11.2 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| seaborn | 0.13.2 |

## 4. Dataset sources

Both datasets are loaded from scikit-learn, so no download is needed (https://scikit-learn.org/stable/datasets/toy_dataset.html):

- Breast Cancer: `sklearn.datasets.load_breast_cancer` (Breast Cancer Wisconsin Diagnostic dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17)
- Wine: `sklearn.datasets.load_wine` (Wine dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/109)

## 5. Repository link

https://github.com/akYavuz/ml-project/tree/main

## 6. Video link

https://youtu.be/GOql-7WvIyo
