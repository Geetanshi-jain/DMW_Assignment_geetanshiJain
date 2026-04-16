🛒 Data Mining & Warehousing (DMW) — Market Basket Analysis

Dataset: Instacart Market Basket Analysis (Kaggle)
Techniques: Data Preprocessing · Association Rules · K-Means · DBSCAN · Hierarchical Clustering

📌 Project Overview

In this project, we analyze Instacart grocery order data to understand purchasing behavior.

The main objectives are:

Identify which products are frequently purchased together (Association Rules)
Group customers based on similar buying patterns (Clustering)
🗂️ Dataset Structure
instacart-market-basket-analysis/
├── orders.csv                  ← Order information (user_id, day, hour)
├── order_products__prior.csv  ← Products included in each order
└── products.csv               ← Product names and IDs

Merge Flow:

order_products__prior.csv ──┐
                            ├──► MERGE on product_id ──► MERGE on order_id ──► Final DataFrame
products.csv ───────────────┘
                           orders.csv
🔄 Project Pipeline
┌─────────────────────────────────────────────────────────────────────┐
│                         PROJECT PIPELINE                            │
├──────────┬──────────────┬──────────────┬─────────────┬─────────────┤
│  STEP 1  │    STEP 2    │    STEP 3    │   STEP 4    │   STEP 5    │
│          │              │              │             │             │
│  Load &  │  Clean &     │  Create      │  Apriori    │ Clustering  │
│  Merge   │  Preprocess  │  Basket      │  Algorithm  │ K-Means /   │
│  Data    │  Data        │  Matrix      │  (MBA)      │ DBSCAN      │
└──────────┴──────────────┴──────────────┴─────────────┴─────────────┘
📊 Step-by-Step Walkthrough
Step 1: Dataset Download & Load
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")

orders = pd.read_csv(f'{path}/orders.csv', nrows=10000)
order_products = pd.read_csv(f'{path}/order_products__prior.csv', nrows=10000)
products = pd.read_csv(f'{path}/products.csv')

⚡ nrows=10000 is used to speed up processing during experimentation.

Step 2: Data Cleaning
Operation	Code	Purpose
Missing Values Check	data.isnull().sum()	Identify null values
Drop Nulls	data.dropna()	Remove incomplete rows
Remove Duplicates	data.drop_duplicates()	Eliminate repeated entries
Select Columns	data[['order_id', 'user_id', ...]]	Keep only relevant columns
Step 3: Basket Matrix Creation
Orders (rows)  ×  Products (columns)

┌──────────┬──────┬──────────┬───────────┬─────────┐
│ order_id │ Milk │ Banana   │ Yogurt    │ Bread   │
├──────────┼──────┼──────────┼───────────┼─────────┤
│ 100001   │ 1    │ 1        │ 0         │ 1       │
│ 100002   │ 0    │ 1        │ 1         │ 0       │
│ 100003   │ 1    │ 0        │ 1         │ 1       │
└──────────┴──────┴──────────┴───────────┴─────────┘

1 = Product is present in the order  
0 = Product is NOT present in the order
Step 4: Market Basket Analysis — Apriori Algorithm

Key Metrics Explained:

┌─────────────┬──────────────────────────────────────────────────────┐
│ Metric      │ Meaning                                              │
├─────────────┼──────────────────────────────────────────────────────┤
│ Support     │ Percentage of orders containing the itemset          │
│ Confidence  │ Probability of buying B given that A is purchased    │
│ Lift        │ Strength of association (>1 indicates positive link) │
└─────────────┴──────────────────────────────────────────────────────┘

Example Rule:

{Banana} ──► {Yogurt}

Support   = 0.05  → Present in 5% of all orders  
Confidence = 0.60 → 60% of banana buyers also buy yogurt  
Lift       = 1.8  → 1.8 times more likely than random chance
Step 5: Clustering Algorithms
5.1 K-Means Clustering
     Cluster 0                     Cluster 1
┌─────────────────┐          ┌─────────────────┐
│  Avocado        │          │  Milk           │
│  Spinach        │          │  Bread          │
│  Blueberries    │          │  Cheese         │
│                 │          │                 │
│ "Healthy        │          │ "Dairy &        │
│  Shoppers"      │          │  Staples Buyers"│
└─────────────────┘          └─────────────────┘
        ↑                          ↑
   Centroid 1                 Centroid 2

n_clusters=2, random_state=42 is used for reproducibility.

5.2 DBSCAN Clustering
●●●        ← Cluster 1 (dense region)
●●●●●
●●●        ★ ← Noise / Outlier (-1 label)

            ◆◆◆◆ ← Cluster 2 (another dense region)
            ◆◆◆
Parameter	Value	Meaning
eps	0.5	Maximum distance for neighborhood
min_samples	1	Minimum points to form a cluster
Label -1	—	Noise points (not part of any cluster)
5.3 Agglomerative Hierarchical Clustering
Step 1: Each order is its own cluster
[A]  [B]  [C]  [D]

Step 2: Merge closest clusters
[A+B]  [C]  [D]

Step 3: Merge again
[A+B+C]  [D]

Step 4: Final merge
[A+B+C+D]

A dendrogram helps decide the optimal number of clusters.

📈 Visualizations in the Notebook
#	Chart	Description
1	Bar Chart	Top 10 most purchased products
2	Histogram + KDE	Distribution of items per order
3	Bar Chart	Average number of items per order
4	Count Plot	DBSCAN cluster distribution
🧰 Libraries Used
pandas          # Data manipulation
matplotlib      # Plotting
seaborn         # Visualization
mlxtend         # Apriori & Association Rules
sklearn         # KMeans, DBSCAN
kagglehub       # Dataset download
▶️ How to Run
# Step 1: Install dependencies
pip install pandas matplotlib seaborn mlxtend scikit-learn kagglehub

# Step 2: Open notebook
jupyter notebook DMW__1_.ipynb

# Step 3: Run all cells
(Kernel → Restart & Run All)
📂 Project Structure
DMW_Project/
├── DMW__1_.ipynb     ← Main Jupyter Notebook
└── README.md         ← Documentation
💡 Key Takeaways
Apriori helps identify frequently co-purchased products → useful for recommendation systems like “People also buy”
K-Means groups customers based on similar purchasing behavior
DBSCAN detects unusual or outlier shopping patterns
Hierarchical Clustering provides a tree-based structure of customer behavior
