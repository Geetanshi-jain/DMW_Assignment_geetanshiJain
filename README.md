# 🛒 Data Mining & Warehousing (DMW) — Market Basket Analysis

> **Dataset:** Instacart Market Basket Analysis (Kaggle)  
> **Techniques:** Data Preprocessing · Association Rules · K-Means · DBSCAN · Hierarchical Clustering

---Which customers have similar buying patterns (Clustering)

## 📌 Project Overview

In this project, we analyze Instacart grocery order data. The goal is to determine:
- Which products are frequently purchased together (**Association Rules**)
- Which customers have similar buying patterns (**Clustering**)

---

## 🗂️ Dataset Structure

```
instacart-market-basket-analysis/
├── orders.csv              ← Order info (user_id, day, hour)
├── order_products__prior.csv ← Which products were in each order
└── products.csv            ← Product names & IDs
```

**Merge Flow:**

```
order_products__prior.csv ──┐
                             ├──► MERGE on product_id ──► MERGE on order_id ──► Final DataFrame
products.csv ───────────────┘                             orders.csv
```

---

## 🔄 Project Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROJECT PIPELINE                            │
├──────────┬──────────────┬──────────────┬─────────────┬─────────────┤
│  STEP 1  │    STEP 2    │    STEP 3    │   STEP 4    │   STEP 5    │
│          │              │              │             │             │
│  Load &  │  Clean &     │  Create      │  Apriori    │ Clustering  │
│  Merge   │  Preprocess  │  Basket      │  Algorithm  │ K-Means /   │
│  Data    │  Data        │  Matrix      │  (MBA)      │ DBSCAN      │
└──────────┴──────────────┴──────────────┴─────────────┴─────────────┘
```

---

## 📊 Step-by-Step Walkthrough

### Step 1: Dataset Download & Load

```python
import kagglehub
path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")

orders = pd.read_csv(f'{path}/orders.csv', nrows=10000)
order_products = pd.read_csv(f'{path}/order_products__prior.csv', nrows=10000)
products = pd.read_csv(f'{path}/products.csv')
```
> ⚡ `nrows=10000` It is used to make the processing faster.

---

### Step 2: Data Cleaning

| Operation | Code | Purpose |
|-----------|------|---------|
| Missing Values Check | `data.isnull().sum()` |find null values  |
| Drop Nulls | `data.dropna()` | remove  Incomplete rows |
| Remove Duplicates | `data.drop_duplicates()` | Repeated entries creation|
| Select Columns | `data[['order_id', 'user_id', ...]]` | Keep only the necessary columns. |

---

### Step 3: Basket Matrix Creation

```
Orders (rows)  ×  Products (columns)
┌──────────┬──────┬──────────┬───────────┬─────────┐
│ order_id │ Milk │  Banana  │  Yogurt   │  Bread  │
├──────────┼──────┼──────────┼───────────┼─────────┤
│  100001  │  1   │    1     │     0     │    1    │
│  100002  │  0   │    1     │     1     │    0    │
│  100003  │  1   │    0     │     1     │    1    │
└──────────┴──────┴──────────┴───────────┴─────────┘
       1 = Product is in the order
       0 = Product is NOT in the order
```

---

### Step 4: Market Basket Analysis — Apriori Algorithm

**Key Metrics Explained:**

```
┌─────────────┬───────────────────────────────────────────────────────────┐
│   Metric    │                      Meaning (Hindi)                      │
├─────────────┼───────────────────────────────────────────────────────────┤
│  Support    │ What percentage of orders contained this itemset?                 │
│  Confidence │ If A is purchased, what is the probability that B will also be purchased?        │
│  Lift       │ Is the co-occurrence of A and B greater than what would be expected by chance? (>1 = yes)│
└─────────────┴───────────────────────────────────────────────────────────┘
```

**Example Rule:**
```
{Banana} ──► {Yogurt}
  Support   = 0.05  → 5% of orders contain both items
  Confidence = 0.60  →  60% chance: customers who buy bananas also buy yogurt
  Lift       = 1.8   →  1.8× more likely than random chance
```

---

### Step 5: Clustering Algorithms

#### 5.1 K-Means Clustering

```
         Cluster 0                    Cluster 1
    ┌─────────────────┐          ┌─────────────────┐
    │  🥑 Avocado     │          │  🥛 Milk        │
    │  🥬 Spinach     │          │  🍞 Bread       │
    │  🫐 Blueberries │          │  🧀 Cheese      │
    │                 │          │                 │
    │  "Healthy       │          │  "Dairy &       │
    │   Shoppers"     │          │   Staples       │
    │                 │          │   Buyers"       │
    └─────────────────┘          └─────────────────┘
              ↑                          ↑
         Centroid 1                 Centroid 2
```

> `n_clusters=2`, `random_state=42` It is used to ensure reproducibility.

#### 5.2 DBSCAN Clustering

```
  ●●●          ← Cluster 1 (dense region)
 ●●●●●
  ●●●

        ★      ← Noise / Outlier (-1 label)

           ◆◆◆◆  ← Cluster 2 (another dense region)
            ◆◆◆
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `eps` | 0.5 | Maximum distance between two points to be neighbors |
| `min_samples` | 1 | Minimum points to form a dense region |
| Label `-1` | — | Noise points (not belonging to any cluster) |

#### 5.3 Agglomerative Hierarchical Clustering (Explained)

```
Step 1: Each order is its own cluster
[A] [B] [C] [D]

Step 2: Merge the two closest clusters
[A+B] [C] [D]

Step 3: Merge again
[A+B+C] [D]

Step 4: Final merge
[A+B+C+D]

We use a dendrogram to decide how many clusters are needed
```

---

## 📈 Visualizations in the Notebook

| # | Chart | Kya dikhata hai |
|---|-------|-----------------|
| 1 | 📊 Bar Chart | Top 10 most purchased products |
| 2 | 📉 Histogram + KDE | Distribution of items added to cart |
| 3 | 📊 Bar Chart | Average number of items per order |
| 4 | 📊 Count Plot | DBSCAN cluster distribution |

---

## 🧰 Libraries Used

```python
pandas          # Data manipulation
matplotlib      # Plotting
seaborn         # Beautiful visualizations
mlxtend         # Apriori & Association Rules
sklearn         # KMeans, DBSCAN clustering
kagglehub       # Dataset download
```

---

## ▶️ How to Run

```bash
# Step 1: Install dependencies
pip install pandas matplotlib seaborn mlxtend scikit-learn kagglehub

# Step 2: Open notebook
jupyter notebook DMW__1_.ipynb

# Step 3: Run all cells (Kernel → Restart & Run All)
```

---

## 📂 Project Structure

```
DMW_Project/
├── DMW__1_.ipynb     ← Main Jupyter Notebook
└── README.md         ← Yeh file (documentation)
```

---

## 💡 Key Takeaways

- **Apriori** We use this to find which products are frequently purchased together → useful for “People also buy…” recommendations.
- **K-Means** It divides customers into similar buying groups.
- **DBSCAN** It detects unusual shopping patterns (outliers).
- **Hierarchical Clustering** It creates a hierarchy of buying behavior.

---

## DMW Assignment 2026 -Geetanshi jain 
