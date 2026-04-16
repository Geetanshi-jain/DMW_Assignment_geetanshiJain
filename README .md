# 🛒 Data Mining & Warehousing (DMW) — Market Basket Analysis

> **Dataset:** Instacart Market Basket Analysis (Kaggle)  
> **Techniques:** Data Preprocessing · Association Rules · K-Means · DBSCAN · Hierarchical Clustering

---

## 📌 Project Overview

Is project mein hum **Instacart** ke grocery orders ka analysis karte hain. Goal yeh hai ki pata karein:
- Kaunse products ek saath kharide jaate hain? (**Association Rules**)
- Kaunse customers similar buying patterns rakhte hain? (**Clustering**)

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
> ⚡ `nrows=10000` isliye use kiya gaya hai taaki processing fast ho.

---

### Step 2: Data Cleaning

| Operation | Code | Purpose |
|-----------|------|---------|
| Missing Values Check | `data.isnull().sum()` | Null values dhundna |
| Drop Nulls | `data.dropna()` | Incomplete rows remove karna |
| Remove Duplicates | `data.drop_duplicates()` | Repeated entries hatana |
| Select Columns | `data[['order_id', 'user_id', ...]]` | Sirf zaroori columns rakhna |

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
│  Support    │ Kitne % orders mein yeh itemset tha?                      │
│  Confidence │ Agar A liya, toh B lene ki probability kitni hai?         │
│  Lift       │ A aur B ka saath hona, chance se zyada hai kya? (>1 = ha) │
└─────────────┴───────────────────────────────────────────────────────────┘
```

**Example Rule:**
```
{Banana} ──► {Yogurt}
  Support   = 0.05  →  5% orders mein dono hain
  Confidence = 0.60  →  60% chance: jo banana leta hai, yogurt bhi leta hai
  Lift       = 1.8   →  1.8x zyada likely than random
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

> `n_clusters=2`, `random_state=42` use kiya gaya reproducibility ke liye.

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
| Label `-1` | — | Noise points (kisi cluster mein nahi) |

#### 5.3 Agglomerative Hierarchical Clustering (Explained)

```
Step 1: Har order apna khud ka cluster hai
  [A]  [B]  [C]  [D]

Step 2: Sabse paas wale 2 clusters merge karo
  [A+B]  [C]  [D]

Step 3: Dobara merge karo
  [A+B+C]  [D]

Step 4: Final merge
  [A+B+C+D]

Dendrogram se hum decide karte hain → kitne clusters chahiye
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

- **Apriori** se hum pata karte hain ki kaunse products aksar saath mein kharide jaate hain → useful for "People also buy..." recommendations
- **K-Means** customers ko similar buying groups mein divide karta hai
- **DBSCAN** unusual shopping patterns (outliers) ko detect karta hai
- **Hierarchical Clustering** buying behavior ka ek hierarchy banata hai

---

> 📝 *Yeh notebook Data Mining & Warehousing (DMW) course ke practical component ke liye banaya gaya hai.*
