# Chapter 2: End-to-End Machine Learning Project

**Purpose**: This is your complete playbook for executing ANY machine learning project from start to finish. Use this as a checklist and reference guide for real-world projects.


## Table of Contents
1. [The Complete ML Project Workflow](#the-complete-ml-project-workflow)
2. [Step 1: Look at the Big Picture](#step-1-look-at-the-big-picture)
3. [Step 2: Get the Data](#step-2-get-the-data)
4. [Step 3: Explore and Visualize the Data](#step-3-explore-and-visualize-the-data)
5. [Step 4: Prepare the Data](#step-4-prepare-the-data)
6. [Step 5: Select and Train Models](#step-5-select-and-train-models)
7. [Step 6: Fine-Tune Your Model](#step-6-fine-tune-your-model)
8. [Step 7: Present Your Solution](#step-7-present-your-solution)
9. [Step 8: Launch, Monitor, and Maintain](#step-8-launch-monitor-and-maintain)
10. [Code Patterns Library](#code-patterns-library)
11. [Common Pitfalls & Solutions](#common-pitfalls-and-solutions)
12. [Quick Reference Tables](#quick-reference-tables)


## The Complete ML Project Workflow

### The 8-Step Framework

```
1. Look at the Big Picture
   ↓
2. Get the Data
   ↓
3. Explore & Visualize
   ↓
4. Prepare the Data
   ↓
5. Select & Train Models
   ↓
6. Fine-Tune Your Model
   ↓
7. Present Your Solution
   ↓
8. Launch, Monitor & Maintain
```

**⚠️ CRITICAL RULE**: Create your test set in Step 2 and NEVER look at it until Step 6!


## Step 1: Look at the Big Picture

### Frame the Problem

#### Questions to Ask Before You Start

```
□ What is the business objective?
  → How will the model be used?
  → What's the impact if it fails?
  
□ What is the current solution (if any)?
  → What's the baseline performance?
  → What are the pain points?
  
□ What type of ML problem is this?
  → Supervised vs Unsupervised
  → Classification vs Regression
  → Batch vs Online learning
  
□ What are the constraints?
  → Real-time predictions needed?
  → How much data is available?
  → Computational resources?
```

### Problem Type Decision Tree

```
START: What are you trying to predict?

├─ Do you have labeled data?
│  │
│  ├─ NO → Unsupervised Learning
│  │       ├─ Looking for groups? → Clustering
│  │       ├─ Reducing dimensions? → Dimensionality Reduction
│  │       └─ Finding anomalies? → Anomaly Detection
│  │
│  └─ YES → Supervised Learning
│           │
│           ├─ Predicting a CATEGORY? → Classification
│           │   ├─ Two classes → Binary Classification
│           │   └─ Multiple classes → Multiclass Classification
│           │
│           └─ Predicting a VALUE? → Regression
│               ├─ One value → Univariate Regression
│               └─ Multiple values → Multivariate Regression
│
└─ Is there continuous data flow?
    │
    ├─ YES → Consider Online Learning
    │         (updates model incrementally)
    │
    └─ NO → Use Batch Learning
              (trains on all data at once)
```

### Determining Your ML Task Type

**Example: Given California Housing Prices**

> #### Analysis:  
> - [x] We have labeled data (house prices for each district)  
> - [x] We want to predict a value (median house price)  
> - [x] We use multiple features (income, location, etc.)  
> - [x] We predict one value per district  
> - [x] No continuous data stream  
> - [x] Data fits in memory  
>  
> **Conclusion**: Supervised, Multiple Regression, Univariate, Batch Learning


### Select Performance Measure

#### For Regression Problems

**Root Mean Square Error (RMSE)**
- Most common for regression
- Gives higher weight to large errors
- Sensitive to outliers

```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
```

**Formula**:
```
RMSE = √(1/m × Σ(predicted_i - actual_i)²)
```

**Mean Absolute Error (MAE)**
- Less sensitive to outliers
- All errors weighted equally
- Use when you have many outliers

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

**Formula**:
```
MAE = 1/m × Σ|predicted_i - actual_i|
```

#### Decision: RMSE vs MAE

| Use RMSE When | Use MAE When |
|--------------|--------------|
| Errors are normally distributed | Many outliers present |
| Large errors are particularly bad | All errors equally important |
| Standard choice for most cases | Want robust metric |

### Understanding Norms (Distance Measures)

```
ℓ₀ norm: Number of non-zero elements
ℓ₁ norm: Sum of absolute values (Manhattan distance)
ℓ₂ norm: Square root of sum of squares (Euclidean distance)
ℓ∞ norm: Maximum absolute value

Higher norm index → More focus on large values
```

### Check Your Assumptions

**Critical Questions**:

> #### Example scenario:
> Your model outputs prices → feeds into downstream system
>  
> **Assumption**: Downstream system uses exact prices  
> **Reality check**: Does it actually just need price categories?  
> If downstream converts prices to ["cheap", "medium", "expensive"]:  
> > You should be doing CLASSIFICATION, not regression!  
> > Would save months of work!
> 
> ✅ ALWAYS verify assumptions with stakeholders EARLY


## Step 2: Get the Data

### Where to Find Data

**Popular Repositories**:
- OpenML.org
- Kaggle.com
- PapersWithCode.com
- UC Irvine ML Repository
- AWS Public Datasets
- TensorFlow Datasets

**Meta Portals**:
- DataPortals.org
- OpenDataMonitor.eu

### Download Data Programmatically

**Why automate?**
- Data may update regularly
- Need to install on multiple machines
- Reproducibility
- Part of production pipeline

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_data(url, dataset_path, filename):
    """
    Download and extract data if not already present
    
    Args:
        url: URL to download from
        dataset_path: Local directory to store data
        filename: Name of the file/archive
    
    Returns:
        DataFrame with loaded data
    """
    # Create directory if it doesn't exist
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    
    # Download if not present
    file_path = Path(dataset_path) / filename
    if not file_path.is_file():
        urllib.request.urlretrieve(url, file_path)
    
    # Extract if compressed
    if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
        with tarfile.open(file_path) as tar:
            tar.extractall(path=dataset_path)
    
    # Load and return data
    csv_path = Path(dataset_path) / "data.csv"  # Adjust as needed
    return pd.read_csv(csv_path)

# Usage
url = "https://example.com/data.tgz"
data = load_data(url, "datasets", "data.tgz")
```

### Initial Data Inspection

```python
# 1. Look at first few rows
data.head()

# 2. Get dataset info
data.info()
# Shows:
# - Number of entries
# - Column names and types
# - Non-null counts (missing values!)
# - Memory usage

# 3. Check for categorical features
data['categorical_column'].value_counts()

# 4. Statistical summary
data.describe()
# Shows: count, mean, std, min, quartiles, max

# 5. Check for missing values
data.isnull().sum()

# 6. Check data types
data.dtypes
```

### Quick Visualization

```python
import matplotlib.pyplot as plt

# Histograms for all numerical features
data.hist(bins=50, figsize=(12, 8))
plt.tight_layout()
plt.show()
```

**What to look for in histograms**:
- Capped/truncated values (might affect target!)
- Different scales (will need scaling)
- Skewed distributions (may need transformation)
- Outliers

### Create Test Set - DO THIS NOW!

**⚠️ CRITICAL**: Create test set BEFORE exploring data further!

**Why?**
- Prevents data snooping bias
- Your brain will find patterns if you look
- These patterns won't generalize
- You'll overfit to the test set

#### Method 1: Simple Random Split

```python
from sklearn.model_selection import train_test_split

# Basic split
train_set, test_set = train_test_split(
    data, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)
```

#### Method 2: Hash-Based Split (for stable splits)

**Use when**: Dataset gets updated regularly

```python
from zlib import crc32
import numpy as np

def is_id_in_test_set(identifier, test_ratio):
    """Check if ID should be in test set using hash"""
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    """Split data using hash of ID column"""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# If no ID column, use row index
data_with_id = data.reset_index()  # adds 'index' column
train_set, test_set = split_data_with_id_hash(data_with_id, 0.2, "index")

# Or create ID from stable features
data["id"] = data["longitude"] * 1000 + data["latitude"]
train_set, test_set = split_data_with_id_hash(data, 0.2, "id")
```

**Benefits of hash-based splitting**:
- Same instances always in same set
- New data properly distributed
- Works even after data updates

#### Method 3: Stratified Split (RECOMMENDED)

**Use when**: Some feature is very important for prediction

**Why?** Ensures test set is representative of all important groups

```python
from sklearn.model_selection import train_test_split

# Example: Income is important, so create income categories
data["income_cat"] = pd.cut(
    data["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Stratified split - maintains income distribution
train_set, test_set = train_test_split(
    data,
    test_size=0.2,
    stratify=data["income_cat"],  # Key parameter!
    random_state=42
)

# Remove the temporary category column
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

**Verify stratification worked**:
```python
# Check proportions in test set
test_set["income_cat"].value_counts() / len(test_set)

# Compare with overall dataset
data["income_cat"].value_counts() / len(data)

# Should be nearly identical!
```

#### Advanced: Multiple Stratified Splits

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(
    n_splits=10,        # Create 10 different splits
    test_size=0.2,
    random_state=42
)

strat_splits = []
for train_index, test_index in splitter.split(data, data["income_cat"]):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]
    strat_splits.append([strat_train_set, strat_test_set])

# Use first split or evaluate on all for better performance estimate
train_set, test_set = strat_splits[0]
```

### Test Set Size Guidelines

| Dataset Size | Test Set Size |
|-------------|---------------|
| < 10,000 samples | 20-30% |
| 10k - 100k samples | 10-20% |
| 100k - 1M samples | 5-10% |
| > 1M samples | 1-5% |

**Rule of thumb**: Test set should have enough samples to give you confidence in performance estimate (usually 1,000+ samples minimum).


## Step 3: Explore and Visualize the Data

### Golden Rules

```
✅ ONLY explore the TRAINING set
✅ Create a copy before experimenting
✅ Document all insights
❌ NEVER touch the test set
❌ Don't make decisions based on patterns you "see" without verification
```

### Make a Working Copy

```python
# Always work on a copy!
data = train_set.copy()
```

### Geographic Data Visualization

**For data with location information**:

```python
import matplotlib.pyplot as plt

# Basic scatterplot
data.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True
)
plt.show()

# Better: Show density with alpha
data.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    alpha=0.2  # Makes high-density areas visible
)
plt.show()

# Even better: Encode more information
data.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=data["population"] / 100,        # Size = population
    c="median_house_value",             # Color = price
    cmap="jet",                         # Blue (low) to red (high)
    colorbar=True,
    alpha=0.4,
    figsize=(10, 7)
)
plt.show()
```

**Insights from visualization**:
- Geographic clusters (cities, regions)
- Price patterns (coastal vs inland)
- Population density correlation
- Outliers and anomalies

### Correlation Analysis

#### Calculate Correlation Matrix

```python
# Compute all pairwise correlations
corr_matrix = data.corr()

# Look at correlations with target
corr_matrix["target_column"].sort_values(ascending=False)
```

**Understanding correlation values**:
```
+1.0 : Perfect positive correlation
+0.7 to +1.0 : Strong positive
+0.3 to +0.7 : Moderate positive
-0.3 to +0.3 : Weak/no correlation
-0.7 to -0.3 : Moderate negative
-1.0 to -0.7 : Strong negative
-1.0 : Perfect negative correlation
```

**⚠️ Important**: Correlation only measures LINEAR relationships!

#### Visualize Correlations

```python
from pandas.plotting import scatter_matrix

# Select most promising attributes
attributes = ["target", "feature1", "feature2", "feature3"]

scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

# Zoom in on most promising relationship
data.plot(
    kind="scatter",
    x="most_correlated_feature",
    y="target",
    alpha=0.1,
    grid=True
)
plt.show()
```

**What to look for**:
- Strong linear trends
- Non-linear patterns (correlation won't catch these!)
- Outliers and caps
- Horizontal/vertical lines (data artifacts)

### Feature Engineering: Experiment with Combinations

**Goal**: Create features that better correlate with target

```python
# Example: Ratios often more informative than raw numbers
data["rooms_per_house"] = data["total_rooms"] / data["households"]
data["bedrooms_ratio"] = data["total_bedrooms"] / data["total_rooms"]
data["people_per_house"] = data["population"] / data["households"]

# Check new correlations
corr_matrix = data.corr()
corr_matrix["target"].sort_values(ascending=False)
```

**Common feature combinations**:
- Ratios (X / Y)
- Differences (X - Y)
- Products (X × Y)
- Aggregations by group
- Time-based features (day of week, month, etc.)

### Exploration Checklist

```
□ Visualized geographic/spatial patterns
□ Checked correlations with target
□ Identified data quirks (caps, outliers, strange patterns)
□ Created and tested new feature combinations
□ Documented insights for later use
□ Identified features with skewed distributions
□ Identified features needing transformation
□ Found any missing values patterns
□ Checked for duplicate rows
```


## Step 4: Prepare the Data

### Why Write Functions Instead of Manual Preparation?

```
✅ Reproducibility - run same transformations on any dataset
✅ Build a library - reuse across projects
✅ Production ready - same code for training and serving
✅ Easy experimentation - try different combinations quickly
✅ Automation - can be part of pipeline
```

### Separate Predictors and Labels

```python
# Create clean copy
data = train_set.copy()

# Separate features and target
X_train = data.drop("target_column", axis=1)
y_train = data["target_column"].copy()
```

### Handle Missing Values

#### Option 1: Manual Pandas Methods

```python
# Option 1: Drop rows with missing values
data.dropna(subset=["column_with_nulls"], inplace=True)

# Option 2: Drop entire column
data.drop("column_with_nulls", axis=1, inplace=True)

# Option 3: Impute with a value
median = data["column_with_nulls"].median()
data["column_with_nulls"].fillna(median, inplace=True)
```

#### Option 2: Scikit-Learn SimpleImputer (RECOMMENDED)

**Why better?**
- Learns values from training set
- Applies same values to validation/test sets
- Works in production pipelines

```python
from sklearn.impute import SimpleImputer

# Create imputer
imputer = SimpleImputer(strategy="median")  # or "mean", "most_frequent", "constant"

# Fit on training data (learns the median of each feature)
imputer.fit(X_train_num)  # Only numerical features for median

# Check learned values
print(imputer.statistics_)

# Transform training data
X_train_imputed = imputer.transform(X_train_num)

# Later: transform test data with SAME learned values
X_test_imputed = imputer.transform(X_test_num)
```

**Imputation strategies**:
```python
# Numerical features
SimpleImputer(strategy="median")      # Robust to outliers
SimpleImputer(strategy="mean")        # If normally distributed
SimpleImputer(strategy="constant", fill_value=0)  # Fill with specific value

# Categorical features
SimpleImputer(strategy="most_frequent")  # Mode
SimpleImputer(strategy="constant", fill_value="missing")
```

#### Advanced Imputation

```python
from sklearn.impute import KNNImputer, IterativeImputer

# KNN Imputer: Replace with mean of k-nearest neighbors
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X_train)

# Iterative Imputer: Train regression model to predict missing values
iter_imputer = IterativeImputer(random_state=42)
X_imputed = iter_imputer.fit_transform(X_train)
```

### Handle Categorical Features

#### Understanding the Problem

```python
# Categorical data is text
data["ocean_proximity"].head()
# ['NEAR BAY', 'INLAND', 'NEAR OCEAN', ...]

# ML algorithms need numbers!
```

#### Option 1: Ordinal Encoding (For Ordered Categories)

**Use when**: Categories have a natural order (bad < average < good)

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
data_encoded = ordinal_encoder.fit_transform(data[["categorical_col"]])

# Check mapping
ordinal_encoder.categories_
```

**⚠️ Problem**: ML algorithms assume nearby values are similar!
- Category 0 and 1 seem more similar than 0 and 4
- Only use for truly ordered categories

#### Option 2: One-Hot Encoding (For Unordered Categories)

**Use when**: Categories have no inherent order

```python
from sklearn.preprocessing import OneHotEncoder

# Create encoder
cat_encoder = OneHotEncoder(
    handle_unknown="ignore",  # Important for production!
    sparse=False              # Return dense array instead of sparse
)

# Fit and transform
data_encoded = cat_encoder.fit_transform(data[["categorical_col"]])

# Check categories
cat_encoder.categories_

# Get feature names
cat_encoder.get_feature_names_out()
```

**Example**:
```
Input: ['INLAND', 'NEAR OCEAN', 'INLAND']

Output (one-hot encoded):
[[0, 1, 0, 0, 0],  # INLAND
 [0, 0, 0, 0, 1],  # NEAR OCEAN
 [0, 1, 0, 0, 0]]  # INLAND
```

**Why OneHotEncoder > pandas get_dummies()**:

```python
# ❌ pandas get_dummies()
pd.get_dummies(df["ocean_proximity"])
# Problem: Doesn't remember categories from training
# Will create different columns for new data!

# ✅ OneHotEncoder
encoder = OneHotEncoder()
encoder.fit_transform(train_data)
# Remembers categories
encoder.transform(new_data)  # Uses SAME categories, handles unknowns
```

**Handling unknown categories**:

```python
# Option 1: Ignore (represent as all zeros)
encoder = OneHotEncoder(handle_unknown="ignore")

# Option 2: Error (safer for debugging)
encoder = OneHotEncoder(handle_unknown="error")  # Default

# Test it
df_test = pd.DataFrame({"ocean_proximity": ["<2H OCEAN"]})  # Unknown!
encoder.transform(df_test)  # All zeros if handle_unknown="ignore"
```

#### Dealing with High Cardinality

**Problem**: Too many categories → too many features after one-hot encoding

**Solutions**:

1. **Replace with numerical feature**  
    Instead of: country_code (195 categories)  
    Use: country_population, country_gdp_per_capita

2. **Group rare categories**  
    Combine categories with < 1% frequency into "other"

3. **Use target encoding (advanced)**  
    Replace category with mean of target for that category

4. **Use embeddings (for neural networks)**  
    Learn low-dimensional representation


### Feature Scaling

**Why scale?**
- ML algorithms perform poorly with different scales
- Gradient descent converges faster
- Distance-based algorithms (KNN, SVM) need it

**⚠️ CRITICAL**: Only fit scalers on training set!

```python
# ❌ WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data including test set!

# ✅ CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training only
X_test_scaled = scaler.transform(X_test)         # Apply same scaling
```

#### Method 1: Min-Max Scaling (Normalization)

**Formula**: `X_scaled = (X - X_min) / (X_max - X_min)`

**Result**: Values in range [0, 1] (or custom range)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))  # or (-1, 1) for neural networks
X_scaled = scaler.fit_transform(X_train)
```

**Pros**:
- Bounded output range
- Preserves zero entries in sparse data

**Cons**:
- Very sensitive to outliers
- Outlier can crush all other values

#### Method 2: Standardization (Z-score Normalization)

**Formula**: `X_scaled = (X - mean) / std`

**Result**: Mean = 0, Standard deviation = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Pros**:
- Less affected by outliers
- No bounded range (neural networks handle this fine)

**Cons**:
- No bounded range (if you need 0-1 range, use MinMaxScaler)

#### When to Use Which?

| Use MinMaxScaler | Use StandardScaler |
|-----------------|-------------------|
| Neural networks with bounded activation (sigmoid, tanh) | Most other algorithms |
| Need values in specific range | Features have outliers |
| Know min/max bounds | Unknown value ranges |
| Image data (pixel values) | General-purpose |

### Feature Transformations

#### Handling Heavy-Tailed Distributions

**Problem**: Features with long tails → most values crushed into small range

**Solution 1: Log Transform**

```python
import numpy as np

# For right-skewed features (long tail to the right)
data["log_feature"] = np.log(data["feature"])  # feature must be > 0

# If feature can be 0 or negative
data["log_feature"] = np.log1p(data["feature"])  # log(1 + x)
```

**When to use**:
- Feature follows power law
- Very long right tail
- Want approximately normal distribution

**Solution 2: Square Root / Power Transform**

```python
# Square root (for moderate skew)
data["sqrt_feature"] = np.sqrt(data["feature"])

# Arbitrary power (0 < power < 1)
data["power_feature"] = data["feature"] ** 0.5
```

#### Bucketizing / Binning

**Use when**: Feature has multimodal distribution or you want to capture thresholds

```python
# Equal-width bins
data["binned"] = pd.cut(
    data["feature"],
    bins=5,                    # Number of bins
    labels=[1, 2, 3, 4, 5]     # Optional: custom labels
)

# Equal-frequency bins (quantiles)
data["binned"] = pd.qcut(
    data["feature"],
    q=5,                       # Number of quantiles
    labels=[1, 2, 3, 4, 5]
)

# Custom bin edges
data["binned"] = pd.cut(
    data["age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["child", "young_adult", "adult", "middle_age", "senior"]
)

# Then one-hot encode the bins!
encoder = OneHotEncoder()
binned_encoded = encoder.fit_transform(data[["binned"]])
```

#### Radial Basis Function (RBF) Features

**Use when**: Want to capture similarity to important values

```python
from sklearn.metrics.pairwise import rbf_kernel

# Add feature measuring similarity to age 35
age_simil_35 = rbf_kernel(
    data[["age"]],
    [[35]],              # Center point
    gamma=0.1            # Controls width of peak
)

# Multiple similarity features
centers = [[20], [35], [50], [65]]
for center in centers:
    col_name = f"age_simil_{center[0]}"
    data[col_name] = rbf_kernel(data[["age"]], [center], gamma=0.1)
```

**Gamma parameter**:
- High gamma → narrow peak (only very close values are similar)
- Low gamma → wide peak (broader range of values are similar)

### Custom Transformers

#### Simple Function Transformer

```python
from sklearn.preprocessing import FunctionTransformer

# Log transformer
log_transformer = FunctionTransformer(
    np.log,
    inverse_func=np.exp,  # For inverse_transform()
    validate=True
)

transformed = log_transformer.fit_transform(data[["feature"]])

# With hyperparameters
rbf_transformer = FunctionTransformer(
    rbf_kernel,
    kw_args=dict(Y=[[35]], gamma=0.1)
)

# Ratio transformer
def ratio_transform(X):
    return X[:, [0]] / X[:, [1]]

ratio_transformer = FunctionTransformer(ratio_transform)
ratios = ratio_transformer.transform(data[["numerator", "denominator"]])
```

#### Full Custom Transformer (Trainable)

**Template**:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, hyperparameter=default_value):
        """
        Initialize with hyperparameters (no *args or **kwargs!)
        """
        self.hyperparameter = hyperparameter
    
    def fit(self, X, y=None):
        """
        Learn parameters from training data
        
        Args:
            X: Training features
            y: Training labels (required in signature even if unused)
        
        Returns:
            self
        """
        X = check_array(X)  # Validate input
        
        # Learn parameters (example: mean)
        self.learned_param_ = X.mean(axis=0)
        
        # Store number of features (required!)
        self.n_features_in_ = X.shape[1]
        
        return self  # Always return self!
    
    def transform(self, X):
        """
        Transform data using learned parameters
        
        Args:
            X: Data to transform
        
        Returns:
            Transformed data
        """
        check_is_fitted(self)  # Ensure fit() was called
        X = check_array(X)
        
        # Verify same number of features
        assert self.n_features_in_ == X.shape[1]
        
        # Apply transformation
        return X - self.learned_param_
    
    def get_feature_names_out(self, names=None):
        """
        Return output feature names (optional but recommended)
        """
        return [f"transformed_{i}" for i in range(self.n_features_in_)]
```

#### Real Example: Cluster Similarity Transformer

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    
    def fit(self, X, y=None, sample_weight=None):
        # Find clusters in training data
        self.kmeans_ = KMeans(
            self.n_clusters,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        # Measure similarity to each cluster center
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Usage
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
similarities = cluster_simil.fit_transform(
    data[["latitude", "longitude"]],
    sample_weight=target_values
)
```

### Transformation Pipelines

#### Why Use Pipelines?

- [x] Ensures correct sequence of transformations
- [x] Prevents data leakage (fit only on training)
- [x] Makes code cleaner and more maintainable
- [x] Easy to experiment with different combinations
- [x] Production-ready (same code for training and inference)


#### Basic Pipeline

```python
from sklearn.pipeline import Pipeline

# Create pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Fit and transform in one call
X_prepared = num_pipeline.fit_transform(X_train)

# Later: transform test set
X_test_prepared = num_pipeline.transform(X_test)
```

#### Pipeline with make_pipeline (Shorter Syntax)

```python
from sklearn.pipeline import make_pipeline

# Automatically names steps
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
# Step names: "simpleimputer", "standardscaler"
```

#### ColumnTransformer: Different Transformations for Different Columns

```python
from sklearn.compose import ColumnTransformer

# Define column groups
num_attribs = ["longitude", "latitude", "housing_median_age", ...]
cat_attribs = ["ocean_proximity"]

# Define pipelines for each type
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Combine everything
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Or drop columns
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
    ("drop_cols", "drop", ["col_to_drop"]),
])

# Or passthrough (keep unchanged)
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
], remainder="passthrough")  # Keep all other columns
```

#### Automatic Column Selection

```python
from sklearn.compose import make_column_selector, make_column_transformer

# Automatically select by type
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
```

#### Complete Preprocessing Pipeline Example

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define helper functions
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

# Pipeline for log-transformed features
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

# Clustering-based features
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

# Default numerical pipeline
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# Categorical pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Combine everything
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder=default_num_pipeline)

# Apply preprocessing
X_prepared = preprocessing.fit_transform(X_train)

# Get feature names
feature_names = preprocessing.get_feature_names_out()
print(feature_names)
```

#### Pipeline Tips

```python
# Access specific step
preprocessing["geo"]  # Returns the cluster_simil transformer

# Access via named_steps
preprocessing.named_steps["geo"]

# Slice pipeline
preprocessing[:-1]  # All steps except last

# Get all steps
preprocessing.steps  # List of (name, estimator) tuples

# Visualize pipeline (in Jupyter)
import sklearn
sklearn.set_config(display="diagram")
preprocessing  # Shows interactive diagram
```

### Target Transformation

**When to transform target**:
- Heavy-tailed distribution (log transform helps)
- Different scale than features
- Want model to predict log(y) instead of y

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

# Method 1: Manual
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y_train.to_frame())

model = LinearRegression()
model.fit(X_train, y_scaled)

# Make predictions
predictions_scaled = model.predict(X_test)
predictions = target_scaler.inverse_transform(predictions_scaled)

# Method 2: TransformedTargetRegressor (EASIER)
model = TransformedTargetRegressor(
    LinearRegression(),
    transformer=StandardScaler()
)

model.fit(X_train, y_train)  # Automatically scales y_train
predictions = model.predict(X_test)  # Automatically inverse-transforms
```


## Step 5: Select and Train Models

### Start Simple

**Strategy**: Train multiple simple models first, then refine

```python
# 1. Always start with a simple baseline
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
baseline_score = dummy.score(X_test, y_test)
print(f"Baseline (just predicting mean): {baseline_score}")

# 2. Linear model
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(X_train, y_train)

# 3. Tree-based model
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(X_train, y_train)

# 4. Ensemble model
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_reg.fit(X_train, y_train)
```

### Evaluate on Training Set

**⚠️ WARNING**: This is NOT enough! Training error can be very misleading.

```python
from sklearn.metrics import mean_squared_error

# Make predictions
predictions = model.predict(X_train)

# Calculate RMSE
rmse = mean_squared_error(y_train, predictions, squared=False)
print(f"Training RMSE: {rmse}")

# Look at some predictions vs actual
predictions[:5].round(-2)  # Round to nearest hundred
y_train.iloc[:5].values
```

**Interpreting training error**:
- Very low (near 0) → Likely overfitting
- Very high → Underfitting (model too simple or bad features)
- Reasonable → Need to check validation error!

### Better Evaluation: Cross-Validation

**Why?**
- Single train/validation split can be misleading
- Get estimate of performance AND uncertainty
- Use all data efficiently

```python
from sklearn.model_selection import cross_val_score

# K-fold cross-validation
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=10,                                  # 10 folds
    scoring="neg_root_mean_squared_error"  # Negative because sklearn wants higher=better
)

# Convert to positive RMSE
rmse_scores = -scores

# Analyze results
import pandas as pd
pd.Series(rmse_scores).describe()
```

**Output interpretation**:
```
count    10.000000
mean     47019.561281   ← Average performance
std       1033.957120   ← How consistent is it?
min      45458.112527
25%      46464.031184
50%      46967.596354   ← Median performance
75%      47325.694987
max      49243.765795
```

**⚠️ Scikit-Learn Quirk**: Cross-validation uses NEGATIVE scores!

```python
# scoring="neg_root_mean_squared_error" returns negative values
# You need to flip the sign:
rmse_scores = -cross_val_score(...)
```

### Compare Multiple Models

```python
import numpy as np
import pandas as pd

def evaluate_model(model, X, y, model_name):
    """Evaluate model using cross-validation"""
    scores = -cross_val_score(
        model, X, y,
        cv=10,
        scoring="neg_root_mean_squared_error"
    )
    return {
        "Model": model_name,
        "Mean RMSE": scores.mean(),
        "Std RMSE": scores.std(),
        "Min RMSE": scores.min(),
        "Max RMSE": scores.max()
    }

# Evaluate all models
results = []
results.append(evaluate_model(lin_reg, X_train, y_train, "Linear Regression"))
results.append(evaluate_model(tree_reg, X_train, y_train, "Decision Tree"))
results.append(evaluate_model(forest_reg, X_train, y_train, "Random Forest"))

# Display results
results_df = pd.DataFrame(results)
results_df.sort_values("Mean RMSE")
```

### Understanding Overfitting vs Underfitting

```python
# Check training error vs validation error

# Training error
train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)

# Validation error (from cross-validation)
val_rmse = -cross_val_score(
    model, X_train, y_train,
    cv=10,
    scoring="neg_root_mean_squared_error"
).mean()

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Difference: {val_rmse - train_rmse:.2f}")

# Interpret:
# train_rmse ≈ 0, val_rmse high → OVERFITTING
# train_rmse high, val_rmse high → UNDERFITTING
# train_rmse low, val_rmse low → Good fit!
```

**Fixing Overfitting**:
1. Simplify model (fewer parameters)
2. Add regularization
3. Get more training data
4. Reduce features (feature selection)
5. Use ensemble methods

**Fixing Underfitting**:
1. Use more complex model
2. Better features (feature engineering)
3. Reduce regularization
4. Train longer (for iterative models)

### Model Selection Guidelines

| Model | When to Use | Pros | Cons |
|-------|------------|------|------|
| **Linear Regression** | Simple baseline, linear relationships | Fast, interpretable | Can't capture non-linear patterns |
| **Ridge/Lasso** | Linear + regularization needed | Prevents overfitting, feature selection (Lasso) | Still linear |
| **Decision Tree** | Non-linear, want interpretability | No scaling needed, interpretable | Easily overfits |
| **Random Forest** | General-purpose, robust | Handles non-linear, robust, good performance | Slower, less interpretable |
| **Gradient Boosting** | Maximum performance needed | Often best performance | Slow, easy to overfit, needs tuning |
| **SVM** | Small-medium datasets, high-dimensional | Effective, memory efficient | Slow on large datasets |
| **Neural Networks** | Very complex patterns, lots of data | Can learn anything | Needs lots of data, slow, hard to interpret |


## Step 6: Fine-Tune Your Model

### Grid Search

**Use when**: Trying combinations of discrete hyperparameter values

```python
from sklearn.model_selection import GridSearchCV

# Build full pipeline
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

# Define parameter grid
param_grid = [
    {
        'preprocessing__geo__n_clusters': [5, 8, 10],
        'random_forest__max_features': [4, 6, 8]
    },
    {
        'preprocessing__geo__n_clusters': [10, 15],
        'random_forest__max_features': [6, 8, 10]
    }
]

# Create grid search
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3,                                  # 3-fold CV
    scoring='neg_root_mean_squared_error',
    return_train_score=True,               # To check for overfitting
    verbose=2                              # Show progress
)

# Run search
grid_search.fit(X_train, y_train)

# Get best parameters
print(grid_search.best_params_)

# Get best estimator (already retrained on full training set)
best_model = grid_search.best_estimator_
```

**Understanding param names**:
```
"preprocessing__geo__n_clusters"
    ↓             ↓        ↓
  Step name   Substep   Hyperparameter

Split on "__" (double underscore):
1. "preprocessing" → Find this step in pipeline
2. "geo" → Find this transformer in ColumnTransformer
3. "n_clusters" → Set this hyperparameter
```

**Viewing results**:
```python
# Convert results to DataFrame
cv_results = pd.DataFrame(grid_search.cv_results_)

# Sort by performance
cv_results.sort_values("mean_test_score", ascending=False)

# Key columns:
# - param_*: Parameter values
# - mean_test_score: Average CV score
# - std_test_score: Standard deviation
# - rank_test_score: Ranking
# - split0_test_score, split1_test_score, ...: Individual fold scores
```

**Grid search tips**:

```python
# 1. Start coarse, then refine
# First pass: Wide range, few values
param_grid = {'C': [0.1, 1, 10, 100]}

# Second pass: Narrow range around best, more values
param_grid = {'C': [80, 90, 100, 110, 120]}

# 2. If best is at boundary, expand search
# If best is 15 (maximum tested):
grid_search.best_params_  # {'n_clusters': 15}
# → Test higher values!

# 3. Cache preprocessing for speed
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", model)
], memory='cache_directory')  # Caches fitted transformers
```

### Randomized Search

**Use when**: 
- Large hyperparameter space
- Continuous hyperparameters
- Limited computational budget

**Why better than grid search for large spaces?**
1. Explores more values per hyperparameter
2. Can run for ANY number of iterations
3. Unaffected by irrelevant hyperparameters

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions
param_distributions = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),  # Uniform integers
    'random_forest__max_features': randint(low=2, high=20),
    'random_forest__n_estimators': randint(low=10, high=200),
    'random_forest__max_depth': randint(low=3, high=30),
    'random_forest__min_samples_split': randint(low=2, high=20),
    'random_forest__bootstrap': [True, False]  # Categorical
}

# Create randomized search
rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distributions,
    n_iter=100,                             # Number of combinations to try
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    verbose=2
)

rnd_search.fit(X_train, y_train)

# Get best parameters
print(rnd_search.best_params_)
best_model = rnd_search.best_estimator_
```

**Available distributions**:
```python
from scipy.stats import randint, uniform, loguniform

# Uniform integers
randint(low=1, high=10)  # 1, 2, 3, ..., 9

# Uniform floats
uniform(loc=0, scale=1)  # 0.0 to 1.0

# Log-uniform (for learning rate, regularization)
loguniform(1e-4, 1e-1)  # 0.0001 to 0.1 (log scale)
```

### Halving Grid/Random Search (Advanced)

**Idea**: Use computational resources more efficiently

**How it works**:
1. Round 1: Evaluate many candidates on small resources (subset of data)
2. Round 2: Best candidates get more resources
3. Round 3: Even fewer candidates, full resources
4. Final: Best candidate trained on full data

```python
from sklearn.experimental import enable_halving_search_cv  # Must import!
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    full_pipeline,
    param_distributions,
    factor=3,           # Each round uses 3x more resources
    cv=3,
    random_state=42
)

halving_search.fit(X_train, y_train)
```

### Ensemble Methods

**Idea**: Combine multiple models for better performance

```python
from sklearn.ensemble import VotingRegressor

# Combine different model types
voting_reg = VotingRegressor([
    ('linear', lin_reg),
    ('forest', forest_reg),
    ('gradient_boosting', gbr)
])

voting_reg.fit(X_train, y_train)

# Prediction is average of all models
predictions = voting_reg.predict(X_test)
```

### Analyze Feature Importances

**For tree-based models**:

```python
# Get feature importances
feature_importances = best_model["random_forest"].feature_importances_

# Get feature names
feature_names = best_model["preprocessing"].get_feature_names_out()

# Create DataFrame
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print(importances_df.head(20))

# Visualize
import matplotlib.pyplot as plt

importances_df.head(20).plot(
    kind='barh',
    x='feature',
    y='importance',
    figsize=(10, 8)
)
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

**What to do with this info**:
- Drop low-importance features (faster training, less overfitting)
- Focus feature engineering on important features
- Understand what model learned

**⚠️ Caution**: Correlated features share importance!

```python
# Automatic feature selection
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    RandomForestRegressor(random_state=42),
    threshold="median"  # Keep features above median importance
)

selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)
```

### Error Analysis

**Goal**: Understand where and why model fails

```python
# Get predictions
predictions = best_model.predict(X_train)

# Calculate errors
errors = predictions - y_train

# Analyze error distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Errors")
plt.show()

# Find worst predictions
error_df = pd.DataFrame({
    'actual': y_train,
    'predicted': predictions,
    'error': errors,
    'abs_error': np.abs(errors)
})

# Largest errors
worst_predictions = error_df.nlargest(20, 'abs_error')
print(worst_predictions)

# Analyze characteristics of worst predictions
# Are they all in a certain category?
# Do they have extreme feature values?
# Are they outliers?
```

**Check performance on subgroups**:

```python
# Example: Performance by price range
price_ranges = pd.cut(y_train, bins=[0, 100000, 200000, 300000, 500000])

for price_range in price_ranges.cat.categories:
    mask = price_ranges == price_range
    if mask.sum() > 0:
        rmse = mean_squared_error(
            y_train[mask],
            predictions[mask],
            squared=False
        )
        print(f"{price_range}: RMSE = {rmse:.2f}, n = {mask.sum()}")
```

**Ensure fairness**: Check performance across sensitive groups
- Geographic regions
- Income levels
- Demographic groups


## Step 7 & 8: Present Your Solution & Deploy Your Model

I don't think this part is really that usefull for practicing. Let's skip it for now.

Later I will add additional Docs & Code-snippets on deploying the Model with better methods(considering CI/CD, Docker, Cloud Services, Monitoring, etc.)


---

## Code Patterns Library

### Pattern 1: Complete End-to-End Pipeline

```python
# Full workflow from data to deployed model

# 1. Load data
data = load_data(url, "datasets", "data.tgz")

# 2. Create test set
train_set, test_set = train_test_split(
    data, test_size=0.2, stratify=data['important_feature_cat'], random_state=42
)

# 3. Separate features and target
X_train = train_set.drop("target", axis=1)
y_train = train_set["target"].copy()
X_test = test_set.drop("target", axis=1)
y_test = test_set["target"].copy()

# 4. Build preprocessing pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

# 5. Build full pipeline with model
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(random_state=42))
])

# 6. Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_features': [4, 6, 8],
    'model__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2
)

grid_search.fit(X_train, y_train)

# 7. Get best model
best_model = grid_search.best_estimator_

# 8. Evaluate on test set
test_predictions = best_model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
print(f"Test RMSE: {test_rmse:.2f}")

# 9. Save model
joblib.dump(best_model, "final_model.pkl")

# 10. Deploy (example: Flask API)
# See deployment section above
```

### Pattern 2: Robust Data Loading

```python
def load_and_validate_data(filepath, required_columns, dtypes=None):
    """
    Load data with validation
    
    Args:
        filepath: Path to data file
        required_columns: List of required column names
        dtypes: Dict of expected dtypes (optional)
    
    Returns:
        DataFrame
    
    Raises:
        ValueError: If validation fails
    """
    try:
        # Load data
        data = pd.read_csv(filepath)
        
        # Check required columns
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check dtypes if specified
        if dtypes:
            for col, expected_dtype in dtypes.items():
                if data[col].dtype != expected_dtype:
                    logging.warning(
                        f"{col}: expected {expected_dtype}, got {data[col].dtype}"
                    )
        
        # Check for completely empty columns
        empty_cols = data.columns[data.isnull().all()].tolist()
        if empty_cols:
            logging.warning(f"Empty columns: {empty_cols}")
        
        # Check for duplicates
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            logging.warning(f"Found {n_duplicates} duplicate rows")
        
        logging.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        return data
    
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise

# Usage
data = load_and_validate_data(
    "data/housing.csv",
    required_columns=["longitude", "latitude", "median_house_value"],
    dtypes={"longitude": np.float64, "latitude": np.float64}
)
```

### Pattern 3: Feature Engineering Pipeline

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer
    """
    def __init__(self, add_ratios=True, add_log=True):
        self.add_ratios = add_ratios
        self.add_log = add_log
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.add_ratios:
            # Add ratio features
            X["rooms_per_house"] = X["total_rooms"] / X["households"]
            X["bedrooms_ratio"] = X["total_bedrooms"] / X["total_rooms"]
            X["people_per_house"] = X["population"] / X["households"]
        
        if self.add_log:
            # Add log-transformed features
            for col in ["total_rooms", "total_bedrooms", "population"]:
                X[f"log_{col}"] = np.log1p(X[col])
        
        return X
    
    def get_feature_names_out(self, names=None):
        feature_names = list(names) if names is not None else []
        
        if self.add_ratios:
            feature_names.extend([
                "rooms_per_house",
                "bedrooms_ratio",
                "people_per_house"
            ])
        
        if self.add_log:
            feature_names.extend([
                "log_total_rooms",
                "log_total_bedrooms",
                "log_population"
            ])
        
        return np.array(feature_names)

# Use in pipeline
pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer(add_ratios=True, add_log=True)),
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor())
])
```

### Pattern 4: Model Evaluation Suite

```python
def evaluate_model_comprehensive(model, X_train, y_train, X_test, y_test):
    """
    Comprehensive model evaluation
    
    Returns:
        dict with all metrics and plots
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    # Cross-validation on training set
    cv_scores = -cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'cv_rmse_mean': cv_scores.mean(),
        'cv_rmse_std': cv_scores.std(),
        'train_rmse': mean_squared_error(y_train, train_pred, squared=False),
        'test_rmse': mean_squared_error(y_test, test_pred, squared=False),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, test_pred)
    }
    
    # Overfitting check
    metrics['overfit_gap'] = metrics['train_rmse'] - metrics['test_rmse']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted (test)
    axes[0, 0].scatter(y_test, test_pred, alpha=0.3)
    axes[0, 0].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Actual vs Predicted (Test Set)')
    
    # Plot 2: Residuals
    test_errors = test_pred - y_test
    axes[0, 1].scatter(test_pred, test_errors, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].set_title('Residual Plot')
    
    # Plot 3: Error distribution
    axes[1, 0].hist(test_errors, bins=50)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Errors')
    
    # Plot 4: CV scores
    axes[1, 1].bar(range(len(cv_scores)), cv_scores)
    axes[1, 1].axhline(y=cv_scores.mean(), color='r', linestyle='--',
                       label=f'Mean: {cv_scores.mean():.2f}')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Cross-Validation Scores')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    print("="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:10.2f}")
    print("="*60)
    
    return metrics, fig

# Usage
metrics, fig = evaluate_model_comprehensive(
    best_model, X_train, y_train, X_test, y_test
)
```


## Common Pitfalls and Solutions

### Pitfall 1: Looking at Test Set Too Early

```python
# ❌ WRONG
data = load_data()
print(data.head())  # Looking at ALL data
X_train, X_test, y_train, y_test = train_test_split(data)

# ✅ CORRECT
data = load_data()
X_train, X_test, y_train, y_test = train_test_split(data, random_state=42)
# NOW explore only X_train
print(X_train.head())
```

### Pitfall 2: Data Leakage in Preprocessing

```python
# ❌ WRONG: Scaling on all data before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT: Fit only on training, transform both
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ EVEN BETTER: Use Pipeline
pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Pitfall 3: Not Handling Categorical Features Properly

```python
# ❌ WRONG: Using get_dummies() in production
train_encoded = pd.get_dummies(train_data)
test_encoded = pd.get_dummies(test_data)
# Problem: Different columns if test has different categories!

# ✅ CORRECT: Use OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(train_data[["category"]])
train_encoded = encoder.transform(train_data[["category"]])
test_encoded = encoder.transform(test_data[["category"]])
```

### Pitfall 4: Ignoring Class Imbalance (for classification)

```python
# For classification problems with imbalance:

# ❌ WRONG: Using simple train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ CORRECT: Use stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Pitfall 5: Overfitting on Hyperparameters

```python
# ❌ WRONG: Tuning on test set
best_params = None
best_score = float('inf')
for params in param_combinations:
    model.set_params(**params)
    model.fit(X_train, y_train)
    score = mean_squared_error(y_test, model.predict(X_test))
    if score < best_score:
        best_score = score
        best_params = params

# ✅ CORRECT: Use cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
# ONLY THEN evaluate on test set ONCE
final_score = best_model.score(X_test, y_test)
```

### Pitfall 6: Not Saving Preprocessing Steps

```python
# ❌ WRONG: Only saving model
joblib.dump(model, "model.pkl")
# Problem: Can't preprocess new data the same way!

# ✅ CORRECT: Save entire pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", model)
])
joblib.dump(pipeline, "full_pipeline.pkl")

# In production:
pipeline = joblib.load("full_pipeline.pkl")
predictions = pipeline.predict(new_data)  # Handles preprocessing automatically
```

### Pitfall 7: Forgetting to Handle Missing Values in Production

```python
# ❌ WRONG: Assuming no missing values in production
model.fit(X_train, y_train)

# ✅ CORRECT: Include imputation in pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestRegressor()
)
pipeline.fit(X_train, y_train)
# Now handles missing values automatically
```

### Pitfall 8: Not Monitoring Model in Production

```python
# ❌ WRONG: Deploy and forget
deploy_model(model)

# ✅ CORRECT: Monitor performance
def make_prediction_with_monitoring(X):
    pred = model.predict(X)
    
    # Log prediction
    log_prediction(X, pred, timestamp=datetime.now())
    
    # Monitor input drift
    if check_data_drift(X, reference_data):
        send_alert("Data drift detected!")
    
    return pred
```


## Quick Reference Tables

### When to Use Which Train/Test Split Method

| Method | Use When | Code |
|--------|----------|------|
| **Simple Random** | Default choice | `train_test_split(X, y, test_size=0.2)` |
| **Stratified** | Important categorical feature | `train_test_split(X, y, stratify=y_cat)` |
| **Time-Based** | Time series data | `X_train = X[:split_date]` |
| **Hash-Based** | Dataset updates regularly | Custom hash function |

### Preprocessing: What Goes Where

| Task | Tool | When to Use |
|------|------|------------|
| **Missing Values - Numerical** | `SimpleImputer(strategy="median")` | Default choice |
| **Missing Values - Categorical** | `SimpleImputer(strategy="most_frequent")` | Categorical data |
| **Scaling** | `StandardScaler()` | Most algorithms |
| **Scaling (bounded)** | `MinMaxScaler()` | Neural networks with sigmoid/tanh |
| **Categorical → Numbers** | `OneHotEncoder()` | Unordered categories |
| **Categorical → Numbers** | `OrdinalEncoder()` | Ordered categories only |
| **Heavy Tail** | `FunctionTransformer(np.log)` | Right-skewed features |
| **Multimodal** | Binning + OneHot | Features with multiple peaks |

### Model Selection Guide

| Data Size | Starting Models | Advanced Models |
|-----------|----------------|-----------------|
| **< 1,000** | Linear Regression, KNN | Random Forest |
| **1k - 10k** | Linear, Random Forest | Gradient Boosting |
| **10k - 100k** | Linear, Random Forest | Gradient Boosting, Neural Nets |
| **> 100k** | SGD, Random Forest | Gradient Boosting, Deep Learning |

### Cross-Validation Fold Selection

| Data Size | Number of Folds | Reason |
|-----------|----------------|---------|
| < 1,000 | 10 or LOO | Need to use data efficiently |
| 1k - 10k | 5-10 | Standard choice |
| 10k - 100k | 3-5 | Faster, still reliable |
| > 100k | 3 or train-val-test | Very fast, enough data |

### Performance Metrics Cheat Sheet

| Problem Type | Primary Metric | Alternative Metrics |
|-------------|----------------|-------------------|
| **Regression** | RMSE | MAE, R², MAPE |
| **Binary Classification** | ROC AUC | Precision, Recall, F1 |
| **Multiclass Classification** | Accuracy (balanced) | Macro/Weighted F1 |
| **Imbalanced Classification** | PR AUC, F1 | Balanced Accuracy |

---

## Project Checklist

### Data Phase
```
□ Downloaded/collected data
□ Created test set and set aside
□ Explored training data only
□ Documented data quirks and issues
□ Identified important features for stratification
```

### Preprocessing Phase
```
□ Handled missing values
□ Encoded categorical features
□ Scaled numerical features
□ Transformed heavy-tailed features
□ Created feature combinations
□ Built reusable pipeline
```

### Modeling Phase
```
□ Started with simple baseline
□ Trained multiple model types
□ Used cross-validation for evaluation
□ Checked for overfitting
□ Selected shortlist of models
```

### Fine-Tuning Phase
```
□ Hyperparameter tuning with GridSearch/RandomSearch
□ Analyzed feature importances
□ Performed error analysis
□ Checked performance on subgroups
□ Ensured no bias/fairness issues
```

---

**End of Chapter 2 Notes**

*Use these notes as your step-by-step guide for every ML project. Adapt the patterns to your specific use case, but follow the general workflow. Good luck!* 🚀

