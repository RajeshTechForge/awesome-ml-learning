# Chapter 3: Classification - Practical Notes

**Purpose**: Use these notes as a step-by-step guide when working on ANY classification problem. Don't just copy-paste blindly—understand the WHY behind each step.


## Table of Contents
1. [Quick Start Workflow](#quick-start-workflow)
2. [Binary Classification](#binary-classification)
3. [Performance Metrics Deep Dive](#performance-metrics-deep-dive)
4. [Choosing the Right Metric](#choosing-the-right-metric)
5. [Multiclass Classification](#multiclass-classification)
6. [Multilabel Classification](#multilabel-classification)
7. [Multioutput Classification](#multioutput-classification)
8. [Error Analysis Framework](#error-analysis-framework)
9. [Code Patterns Library](#code-patterns-library)
10. [Common Pitfalls & Solutions](#common-pitfalls-and-solutions)


## Quick Start Workflow

### Step 0: Before You Begin
```
✓ Understand the problem: Binary, Multiclass, Multilabel, or Multioutput?
✓ Create test set and SET IT ASIDE (don't look at it until the end!)
✓ Check if data is shuffled (most classifiers need this)
✓ Identify if you have class imbalance (affects metric choice)
```

### Step 1: Load and Explore Data
```python
# Pattern for loading data
from sklearn.datasets import fetch_openml  # For standard datasets
# OR for Kaggle/custom data:
import pandas as pd
df = pd.read_csv('your_data.csv')

# Split features and labels
X, y = df.drop('target', axis=1).values, df['target'].values

# Always check shapes
print(f"Features shape: {X.shape}")  # (n_samples, n_features)
print(f"Labels shape: {y.shape}")     # (n_samples,)
print(f"Unique classes: {np.unique(y)}")
```

### Step 2: Train-Test Split
```python
# Method 1: Pre-split dataset (like MNIST)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Method 2: Random split (for most datasets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify maintains class distribution
)
```

### Step 3: Choose Your Classifier
```python
# Start simple, then iterate
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# For large datasets or online learning
clf = SGDClassifier(random_state=42)

# For better performance (but slower)
clf = RandomForestClassifier(random_state=42)
```

### Step 4: Train & Evaluate
```python
# Train
clf.fit(X_train, y_train)

# Evaluate with cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# But ACCURACY IS NOT ENOUGH! See metrics section below.
```


## Binary Classification

### When to Use
- Two-class problems: Spam/Not Spam, Fraud/Legitimate, Positive/Negative
- Can also simplify multiclass problems for learning (e.g., "5 vs Not-5")

### Complete Workflow

#### 1. Create Binary Labels
```python
# Example: Detecting if digit is 5
y_train_5 = (y_train == '5')  # Returns boolean array
y_test_5 = (y_test == '5')

# Check class distribution
print(f"Positive class: {y_train_5.sum()} samples")
print(f"Negative class: {(~y_train_5).sum()} samples")
print(f"Class imbalance ratio: {y_train_5.sum() / len(y_train_5):.2%}")
```

#### 2. Train Classifier
```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Make predictions
predictions = sgd_clf.predict(X_test)
```

#### 3. Understanding Decision Threshold
```python
# Get decision scores (distance from decision boundary)
scores = sgd_clf.decision_function(X_test)

# Default threshold is 0
default_predictions = (scores > 0)

# You can adjust threshold based on your needs
high_precision_predictions = (scores > 3000)  # Higher threshold = higher precision, lower recall
```

**⚠️ KEY CONCEPT**: The threshold determines the precision/recall trade-off. Don't use default blindly!


## Performance Metrics Deep Dive

### The Confusion Matrix

**What it tells you**: How often your classifier confuses each class with another.

```
                    Predicted
                 Negative  Positive
Actual Negative     TN        FP      ← Type I Error (False Alarm)
       Positive     FN        TP      ← Type II Error (Missed Detection)
```

#### Generate Confusion Matrix
```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Get predictions using cross-validation (clean predictions)
y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=3)

# Create confusion matrix
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]
```

#### Visualize Confusion Matrix
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Basic visualization
ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred)
plt.show()

# Normalized by row (shows % of each true class)
ConfusionMatrixDisplay.from_predictions(
    y_train_5, y_train_pred,
    normalize="true",
    values_format=".0%"
)
plt.show()

# Show only errors (set weight=0 for correct predictions)
sample_weight = (y_train_pred != y_train_5)
ConfusionMatrixDisplay.from_predictions(
    y_train_5, y_train_pred,
    sample_weight=sample_weight,
    normalize="true",
    values_format=".0%"
)
plt.show()
```

### Precision, Recall, and F1 Score

#### 📊 The Metrics Triangle

```
                    Precision
                   How accurate are
                  positive predictions?
                        /\
                       /  \
                      /    \
                     /      \
                    /        \
                   /          \
                  /            \
            Recall ----------- F1 Score
        How many positives   Harmonic mean
        did we catch?        (balanced metric)
```

#### Formulas You Need to Remember

```
Precision = TP / (TP + FP)  → "When I predict positive, how often am I right?"
Recall    = TP / (TP + FN)  → "Of all actual positives, how many did I find?"
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Calculate Metrics
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)
f1 = f1_score(y_train_5, y_train_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

### Precision/Recall Curve

**Purpose**: Helps you choose the optimal threshold for your use case.

```python
from sklearn.metrics import precision_recall_curve

# Get decision scores
y_scores = cross_val_predict(
    clf, X_train, y_train_5, cv=3,
    method="decision_function"  # or "predict_proba" for some classifiers
)

# Calculate precision and recall for all thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Subplot 1: Precision and Recall vs Threshold
plt.subplot(1, 2, 1)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

# Subplot 2: Precision vs Recall
plt.subplot(1, 2, 2)
plt.plot(recalls, precisions, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Finding the Right Threshold

**Option 1**: Target specific precision
```python
# Want 90% precision? Find the threshold
target_precision = 0.90
idx = (precisions >= target_precision).argmax()
threshold_90 = thresholds[idx]

# Make predictions with this threshold
y_train_pred_90 = (y_scores >= threshold_90)

# Check actual metrics
print(f"Precision: {precision_score(y_train_5, y_train_pred_90):.3f}")
print(f"Recall: {recall_score(y_train_5, y_train_pred_90):.3f}")
```

**Option 2**: Find best F1 score
```python
# Calculate F1 for all thresholds
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Best F1 threshold: {best_threshold:.2f}")
print(f"F1 Score: {f1_scores[best_idx]:.3f}")
```

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic)**: Plots True Positive Rate vs False Positive Rate

**When to use ROC vs PR Curve?**
- **Use PR Curve**: When positive class is rare OR you care more about false positives
- **Use ROC Curve**: When classes are balanced OR you want a general performance measure

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_train_5, y_scores)

# Calculate AUC (Area Under Curve)
auc = roc_auc_score(y_train_5, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k:', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpreting AUC**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Good
- **AUC = 0.7-0.8**: Fair
- **AUC = 0.5**: Random guessing (useless)
- **AUC < 0.5**: Worse than random (predictions are inverted!)


## Choosing the Right Metric

### Decision Framework

```
START: What is your classification problem?
│
├─ Is the positive class RARE (< 20% of data)?
│  │
│  ├─ YES → Use PR Curve & F1 Score
│  │        (ROC/AUC can be misleading)
│  │
│  └─ NO → Continue...
│
├─ What matters MORE to you?
│  │
│  ├─ False Positives are COSTLY (e.g., spam filter blocking important emails)
│  │  → Optimize for HIGH PRECISION
│  │  → Accept lower recall
│  │
│  ├─ False Negatives are COSTLY (e.g., missing cancer diagnosis)
│  │  → Optimize for HIGH RECALL
│  │  → Accept lower precision
│  │
│  └─ Both are equally important
│     → Use F1 Score or ROC AUC
│
└─ Do you need to compare multiple models?
   → Use ROC AUC for overall performance
   → Use PR Curve for detailed threshold selection
```

### Real-World Examples

| Use Case | Optimize For | Why |
|----------|-------------|-----|
| **Email Spam Filter** | Precision | Don't want to block important emails (false positives) |
| **Cancer Detection** | Recall | Can't miss actual cancer cases (false negatives) |
| **Credit Card Fraud** | Balanced (F1) | Both false alarms and missed fraud are costly |
| **Video Content Moderation (kids)** | Precision | Better to reject good videos than allow bad ones |
| **Shoplifter Detection** | Recall | Better to have false alarms than miss thieves |

### Handling Class Imbalance

**Problem**: When one class dominates (e.g., 90% negative, 10% positive), accuracy is misleading.

```python
# DON'T DO THIS with imbalanced data:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)  # Can be 90% even if model is useless!

# DO THIS instead:
from sklearn.metrics import classification_report, balanced_accuracy_score

# Get comprehensive metrics
print(classification_report(y_test, predictions))

# Use balanced accuracy (averages recall for each class)
balanced_acc = balanced_accuracy_score(y_test, predictions)
```

**Solutions for Imbalance**:
1. **Use appropriate metrics**: Precision, Recall, F1, ROC AUC
2. **Resample data**:
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.under_sampling import RandomUnderSampler
   
   # Oversample minority class
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   
   # Or undersample majority class
   undersampler = RandomUnderSampler(random_state=42)
   X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
   ```
3. **Use class weights**:
   ```python
   # Many classifiers support this
   clf = RandomForestClassifier(class_weight='balanced', random_state=42)
   # OR specify custom weights
   clf = RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=42)
   ```


## Multiclass Classification

### Types of Problems
- **Multiclass**: Each instance belongs to ONE of MANY classes (e.g., digit 0-9)
- Different from binary (2 classes) and multilabel (multiple classes per instance)

### Strategies for Binary Classifiers

#### One-vs-Rest (OvR) / One-vs-All (OvA)
```
Train N binary classifiers (one per class)
Each classifier: "Is it class A?" or "Is it NOT class A?"

Example for digits 0-9:
- Classifier 1: "Is it 0?" → Yes/No
- Classifier 2: "Is it 1?" → Yes/No
- ...
- Classifier 10: "Is it 9?" → Yes/No

Prediction: Choose class with highest confidence score
```

**When to use**: Most cases, especially with large datasets

#### One-vs-One (OvO)
```
Train N×(N-1)/2 classifiers (one for each pair of classes)

Example for digits 0-9:
- Train 45 classifiers (10×9/2)
- Classifier 1: "Is it 0 or 1?"
- Classifier 2: "Is it 0 or 2?"
- ...

Prediction: Run all classifiers, choose class that wins most "duels"
```

**When to use**: When training is slow (e.g., SVM), because each classifier trains on smaller subset

### Implementation

#### Method 1: Automatic (Most Common)
```python
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# Scikit-learn automatically chooses OvR or OvO
# SVC uses OvO
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train, y_train)  # y_train has multiple classes

# SGDClassifier uses OvR
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

# Make predictions
predictions = sgd_clf.predict(X_test)
```

#### Method 2: Force a Strategy
```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Force OvR
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train, y_train)

# Force OvO
ovo_clf = OneVsOneClassifier(SVC(random_state=42))
ovo_clf.fit(X_train, y_train)

# Check number of trained classifiers
print(f"OvR trained {len(ovr_clf.estimators_)} classifiers")  # 10 for digits
print(f"OvO trained {len(ovo_clf.estimators_)} classifiers")  # 45 for digits
```

### Getting Prediction Scores

```python
# Different classifiers have different score methods

# Method 1: decision_function (for SVC, SGDClassifier)
scores = clf.decision_function(X_test)
# Returns: (n_samples, n_classes) array
# Example: [[3.79, 0.73, 6.06, 8.3, -0.29, 9.3, ...]]

# Method 2: predict_proba (for RandomForest, LogisticRegression)
probabilities = clf.predict_proba(X_test)
# Returns: (n_samples, n_classes) array of probabilities
# Example: [[0.01, 0.02, 0.05, 0.10, 0.01, 0.78, ...]]  # sums to 1.0

# Find predicted class
predicted_class = scores.argmax()  # or probabilities.argmax()
class_label = clf.classes_[predicted_class]
```

### Evaluation

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Basic accuracy
scores = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# ⚠️ ALWAYS TRY SCALING for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
scores_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(f"Accuracy (scaled): {scores_scaled.mean():.3f}")
# Often improves by 3-5%!
```

### Multiclass Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Get predictions
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw counts
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0])
axes[0].set_title("Confusion Matrix (Raw Counts)")

# Normalized (percentages)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred,
    normalize="true",
    values_format=".0%",
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix (Normalized)")

plt.tight_layout()
plt.show()
```

**Reading the Matrix**:
- **Rows**: Actual classes
- **Columns**: Predicted classes
- **Diagonal**: Correct predictions
- **Off-diagonal**: Confusion between classes

Example interpretation:
```
If row 5, column 8 shows 10%:
→ "10% of actual 5s were misclassified as 8s"

If you normalize by column:
→ Shows "Of all predictions of class 8, X% were actually class 5"
```


## Error Analysis Framework

### Step-by-Step Process

#### 1. Generate Comprehensive Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Train model and get predictions
y_train_pred = cross_val_predict(clf, X_train_scaled, y_train, cv=3)

# Create figure with multiple views
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# View 1: Raw confusion matrix
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0, 0])
axes[0, 0].set_title("Raw Counts")

# View 2: Normalized by row (recall per class)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred,
    normalize="true",
    values_format=".0%",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Normalized by Row (Recall)")

# View 3: Show only errors (normalized by row)
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred,
    sample_weight=sample_weight,
    normalize="true",
    values_format=".0%",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Errors Only (Normalized by Row)")

# View 4: Normalized by column (precision per class)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred,
    sample_weight=sample_weight,
    normalize="pred",
    values_format=".0%",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Errors Only (Normalized by Column)")

plt.tight_layout()
plt.show()
```

#### 2. Identify Problem Areas

**Questions to ask**:
1. Which classes have lowest recall (row with brightest colors)?
2. Which classes are most confused with each other?
3. Is there a pattern? (e.g., many classes confused as class X)

```python
# Calculate per-class metrics
from sklearn.metrics import classification_report

print(classification_report(y_train, y_train_pred))
```

Output example:
```
              precision    recall  f1-score   support

           0       0.94      0.97      0.96      5923
           1       0.96      0.98      0.97      6742
           2       0.88      0.86      0.87      5958
           3       0.87      0.85      0.86      6131
           ...

    accuracy                           0.91     60000
   macro avg       0.91      0.91      0.91     60000
weighted avg       0.91      0.91      0.91     60000
```

#### 3. Analyze Individual Errors

**Goal**: Understand WHY the model makes mistakes

```python
import numpy as np

# Example: Analyze confusion between classes A and B
class_a, class_b = '3', '5'

# Get images for each confusion type
X_aa = X_train[(y_train == class_a) & (y_train_pred == class_a)]  # Correct A
X_ab = X_train[(y_train == class_a) & (y_train_pred == class_b)]  # A→B error
X_ba = X_train[(y_train == class_b) & (y_train_pred == class_a)]  # B→A error
X_bb = X_train[(y_train == class_b) & (y_train_pred == class_b)]  # Correct B

# Plot in confusion matrix style
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

def plot_samples(images, ax, title, n_samples=5):
    """Plot sample images"""
    if len(images) == 0:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    
    # Select random samples
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        img = images[idx].reshape(28, 28)  # Adjust reshape for your data
        ax_sub = plt.subplot(2, 2, (ax_position_map[ax]), n_samples, 1, i+1)
        ax_sub.imshow(img, cmap='binary')
        ax_sub.axis('off')
    
    ax.set_title(title)

plot_samples(X_aa, axes[0, 0], f'Correctly predicted {class_a}')
plot_samples(X_ab, axes[0, 1], f'{class_a} predicted as {class_b} (ERROR)')
plot_samples(X_ba, axes[1, 0], f'{class_b} predicted as {class_a} (ERROR)')
plot_samples(X_bb, axes[1, 1], f'Correctly predicted {class_b}')

plt.tight_layout()
plt.show()
```

#### 4. Root Cause Analysis

**Common causes of errors**:

1. **Data quality issues**:
   - Mislabeled training data
   - Poor quality images
   - Ambiguous cases (even humans can't tell)

2. **Feature issues**:
   - Model doesn't have the right features to distinguish classes
   - Need feature engineering

3. **Model limitations**:
   - Linear models struggle with similar-looking classes
   - Need more complex model (e.g., neural networks)

4. **Data variability**:
   - Images shifted, rotated, scaled differently
   - Need data augmentation

#### 5. Solutions Matrix

| Problem Identified | Possible Solutions |
|-------------------|-------------------|
| **Classes A and B highly confused** | - Gather more data for these classes<br>- Engineer features to distinguish them<br>- Use ensemble methods |
| **Low recall for rare classes** | - Oversample minority class<br>- Use class weights<br>- Collect more data |
| **Model sensitive to rotation/position** | - Data augmentation<br>- Use rotation-invariant features<br>- Preprocess to center images |
| **Many false positives for class X** | - Adjust decision threshold<br>- Add negative examples<br>- Use cost-sensitive learning |
| **Linear model underperforming** | - Try non-linear models (Random Forest, SVM with RBF kernel)<br>- Add polynomial features<br>- Use deep learning |

### Practical Error Analysis Example

```python
# Complete error analysis workflow

def analyze_errors(clf, X_train, y_train, X_test, y_test):
    """
    Comprehensive error analysis for classification
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get predictions
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    y_test_pred = clf.predict(X_test)
    
    print("="*50)
    print("TRAINING SET ANALYSIS")
    print("="*50)
    print(classification_report(y_train, y_train_pred))
    
    print("\n" + "="*50)
    print("TEST SET ANALYSIS")
    print("="*50)
    print(classification_report(y_test, y_test_pred))
    
    # Find most confused pairs
    cm = confusion_matrix(y_train, y_train_pred)
    classes = np.unique(y_train)
    
    # Zero out diagonal (correct predictions)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    # Find top 5 most confused pairs
    print("\n" + "="*50)
    print("TOP 5 MOST CONFUSED CLASS PAIRS")
    print("="*50)
    
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                confused_pairs.append((cm_errors[i, j], classes[i], classes[j]))
    
    confused_pairs.sort(reverse=True)
    for count, true_class, pred_class in confused_pairs[:5]:
        percentage = 100 * count / cm[true_class, :].sum()
        print(f"{true_class} → {pred_class}: {count} errors ({percentage:.1f}% of all {true_class}s)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ConfusionMatrixDisplay.from_predictions(
        y_train, y_train_pred,
        normalize="true",
        values_format=".2f",
        ax=axes[0]
    )
    axes[0].set_title("Training Set")
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred,
        normalize="true",
        values_format=".2f",
        ax=axes[1]
    )
    axes[1].set_title("Test Set")
    
    plt.tight_layout()
    plt.show()
    
    return confused_pairs[:5]

# Use it
confused = analyze_errors(sgd_clf, X_train_scaled, y_train, X_test_scaled, y_test)
```


## Multilabel Classification

### What is it?
Each instance can belong to MULTIPLE classes simultaneously.

**Examples**:
- Image tagging: [dog, outdoor, sunny]
- Document classification: [sports, breaking_news, video]
- Face recognition: [Alice, Bob] (both people in same photo)

### Implementation

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Example: Create multilabel targets
# Label 1: Is the digit large (≥7)?
y_train_large = (y_train >= '7')

# Label 2: Is the digit odd?
y_train_odd = (y_train.astype('int8') % 2 == 1)

# Combine into multilabel array
y_multilabel = np.c_[y_train_large, y_train_odd]
print(y_multilabel[:5])
# [[False  True]   # digit 5: not large, odd
#  [ True False]   # digit 0: large(?), even
#  ...]

# Train multilabel classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# Make prediction
prediction = knn_clf.predict([some_digit])
print(prediction)  # [[False, True]] → not large, odd
```

### Classifiers Supporting Multilabel

**Native support**:
- KNeighborsClassifier
- RandomForestClassifier
- DecisionTreeClassifier

**Need wrapper**:
- SGDClassifier → Use `OneVsRestClassifier` or `ClassifierChain`
- SVC → Use `OneVsRestClassifier` or `ClassifierChain`

### Evaluation

#### Option 1: Average F1 Score Across Labels

```python
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

# Get predictions
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

# Macro average: unweighted mean (all labels equal)
f1_macro = f1_score(y_multilabel, y_train_knn_pred, average="macro")
print(f"F1 Score (macro): {f1_macro:.3f}")

# Weighted average: weighted by support (number of samples)
f1_weighted = f1_score(y_multilabel, y_train_knn_pred, average="weighted")
print(f"F1 Score (weighted): {f1_weighted:.3f}")

# Per-label scores
f1_per_label = f1_score(y_multilabel, y_train_knn_pred, average=None)
print(f"F1 per label: {f1_per_label}")
```

**When to use which average?**
- **macro**: All labels equally important
- **weighted**: Common labels matter more
- **None**: See individual label performance

#### Option 2: Per-Label Analysis

```python
from sklearn.metrics import classification_report

# Get detailed report
print(classification_report(y_multilabel, y_train_knn_pred,
                          target_names=['large', 'odd']))
```

### Chained Classifiers

**Problem**: Labels might be correlated (e.g., "large digit" affects "odd digit" probability)

**Solution**: Train classifiers in a chain, where each classifier uses predictions from previous ones.

```python
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC

# Create chain: second classifier sees first classifier's prediction
chain_clf = ClassifierChain(
    SVC(),
    cv=3,  # Use cross-validation to get clean predictions for training
    random_state=42
)

# Train on subset (SVC is slow)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# Predict
prediction = chain_clf.predict([some_digit])
print(prediction)  # [[0., 1.]]
```

**How it works**:
1. Train classifier 1 on original features → predicts label 1
2. Train classifier 2 on original features + label 1 prediction → predicts label 2
3. For prediction: Run classifiers in sequence, each using previous predictions

**Trade-offs**:
- ✅ Captures label dependencies
- ✅ Often better performance
- ❌ Slower training and prediction
- ❌ Order of labels matters (try different orders)

### Practical Multilabel Example

```python
# Complete workflow for multilabel classification

def train_multilabel_classifier(X_train, y_multilabel, X_test):
    """
    Train and evaluate multilabel classifier
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import cross_val_predict
    
    # Use Random Forest (natively supports multilabel)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_multilabel)
    
    # Cross-validation predictions
    y_pred = cross_val_predict(clf, X_train, y_multilabel, cv=3)
    
    # Evaluation
    print("Per-label F1 scores:")
    f1_per_label = f1_score(y_multilabel, y_pred, average=None)
    for i, score in enumerate(f1_per_label):
        print(f"  Label {i}: {score:.3f}")
    
    print(f"\nMacro F1: {f1_score(y_multilabel, y_pred, average='macro'):.3f}")
    print(f"Weighted F1: {f1_score(y_multilabel, y_pred, average='weighted'):.3f}")
    
    # Detailed report
    print("\nDetailed Report:")
    print(classification_report(y_multilabel, y_pred))
    
    return clf

# Use it
clf = train_multilabel_classifier(X_train_scaled, y_multilabel, X_test_scaled)
```


## Multioutput Classification

### What is it?
Generalization of multilabel where each label can be multiclass (not just binary).

**Key difference**:
- **Multilabel**: Multiple binary labels → [Yes, No, Yes]
- **Multioutput**: Multiple multiclass labels → [5, 3, 8] (could be any values)

**Example**: Image denoising
- Input: Noisy image
- Output: Clean image (each pixel is a label with 256 possible values: 0-255)

### Implementation: Image Denoising Example

```python
import numpy as np

# Create noisy training data
np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_noisy = X_train + noise

# Create noisy test data
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_noisy = X_test + noise

# Target is the clean image
y_train_clean = X_train
y_test_clean = X_test

# Train classifier
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_noisy, y_train_clean)

# Denoise an image
clean_digit = knn_clf.predict([X_test_noisy[0]])

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Noisy image
axes[0].imshow(X_test_noisy[0].reshape(28, 28), cmap='binary')
axes[0].set_title("Noisy Input")
axes[0].axis('off')

# Cleaned image
axes[1].imshow(clean_digit.reshape(28, 28), cmap='binary')
axes[1].set_title("Model Output")
axes[1].axis('off')

# Original clean image
axes[2].imshow(y_test_clean[0].reshape(28, 28), cmap='binary')
axes[2].set_title("Ground Truth")
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### When to Use Multioutput

**Use cases**:
- Image-to-image tasks (denoising, super-resolution, colorization)
- Sequence-to-sequence (but NLP usually uses specialized models)
- Predicting multiple continuous values (though this is really multioutput regression)

**Note**: The line between classification and regression blurs here. If outputs are continuous, use regression models instead.


## Code Patterns Library

### Pattern 1: Complete Binary Classification Pipeline

```python
def binary_classification_pipeline(X, y, test_size=0.2, target_precision=0.90):
    """
    Complete pipeline for binary classification
    
    Args:
        X: Feature matrix
        y: Binary labels
        test_size: Fraction for test set
        target_precision: Target precision for threshold selection
    
    Returns:
        clf: Trained classifier
        metrics: Dict of performance metrics
    """
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score, precision_recall_curve)
    import numpy as np
    
    # 1. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train classifier
    clf = SGDClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 4. Get decision scores for threshold tuning
    y_scores = cross_val_predict(
        clf, X_train_scaled, y_train, cv=3,
        method="decision_function"
    )
    
    # 5. Find threshold for target precision
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    idx = (precisions >= target_precision).argmax()
    threshold = thresholds[idx]
    
    # 6. Make predictions with optimal threshold
    y_test_scores = clf.decision_function(X_test_scaled)
    y_test_pred = (y_test_scores >= threshold)
    
    # 7. Calculate metrics
    metrics = {
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_scores),
        'threshold': threshold
    }
    
    # 8. Print results
    print("Binary Classification Results")
    print("="*50)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Optimal Threshold: {metrics['threshold']:.2f}")
    
    return clf, scaler, metrics

# Use it
clf, scaler, metrics = binary_classification_pipeline(X, y_binary)
```

### Pattern 2: Complete Multiclass Classification Pipeline

```python
def multiclass_classification_pipeline(X, y, test_size=0.2):
    """
    Complete pipeline for multiclass classification
    
    Args:
        X: Feature matrix
        y: Multiclass labels
        test_size: Fraction for test set
    
    Returns:
        clf: Trained classifier
        metrics: Dict of performance metrics
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    
    # 1. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 4. Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    
    # 5. Test set evaluation
    y_test_pred = clf.predict(X_test_scaled)
    
    # 6. Print results
    print("Multiclass Classification Results")
    print("="*50)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_test_pred))
    
    # 7. Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred,
        normalize="true",
        values_format=".2f",
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    
    plt.tight_layout()
    plt.show()
    
    return clf, scaler

# Use it
clf, scaler = multiclass_classification_pipeline(X, y_multiclass)
```

### Pattern 3: Model Comparison

```python
def compare_classifiers(X_train, y_train, X_test, y_test):
    """
    Compare multiple classifiers
    """
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
    import pandas as pd
    import time
    
    classifiers = {
        'SGD': SGDClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Time training
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Test set performance
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Classifier': name,
            'CV Accuracy': f'{cv_mean:.3f} ± {cv_std:.3f}',
            'Test Accuracy': f'{test_acc:.3f}',
            'Test F1': f'{test_f1:.3f}',
            'Train Time (s)': f'{train_time:.2f}'
        })
    
    # Display results
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("CLASSIFIER COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    
    return df

# Use it
results = compare_classifiers(X_train_scaled, y_train, X_test_scaled, y_test)
```

### Pattern 4: Hyperparameter Tuning

```python
def tune_hyperparameters(X_train, y_train, classifier_type='random_forest'):
    """
    Hyperparameter tuning using GridSearchCV
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    
    if classifier_type == 'random_forest':
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif classifier_type == 'sgd':
        clf = SGDClassifier(random_state=42)
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [1000, 5000]
        }
    
    elif classifier_type == 'svm':
        clf = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        clf, param_grid, cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1  # Use all CPU cores
    )
    
    print(f"Tuning {classifier_type}...")
    grid_search.fit(X_train, y_train)
    
    # Results
    print("\n" + "="*50)
    print("BEST PARAMETERS")
    print("="*50)
    print(grid_search.best_params_)
    print(f"\nBest CV Score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

# Use it
best_clf = tune_hyperparameters(X_train_scaled, y_train, 'random_forest')
```


## Common Pitfalls & Solutions

### Pitfall 1: Not Creating a Test Set First
```python
# ❌ WRONG: Looking at data before creating test set
X, y = load_data()
print(X[:10])  # Oops! Already looked at data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# ✅ CORRECT: Create test set FIRST, then explore
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Now explore only training data
print(X_train[:10])
```

### Pitfall 2: Using Accuracy for Imbalanced Data
```python
# ❌ WRONG: Accuracy on imbalanced data
y_train_5 = (y_train == '5')  # Only 10% are 5s
accuracy = cross_val_score(clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(f"Accuracy: {accuracy.mean():.3f}")  # Looks good but misleading!

# ✅ CORRECT: Use appropriate metrics
from sklearn.metrics import f1_score, roc_auc_score
y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=3)
print(f"F1 Score: {f1_score(y_train_5, y_train_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_train_5, y_scores):.3f}")
```

### Pitfall 3: Forgetting to Scale Features
```python
# ❌ WRONG: Training without scaling
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
scores = cross_val_score(sgd_clf, X_train, y_train, cv=3)
print(f"Accuracy: {scores.mean():.3f}")  # Lower than it could be!

# ✅ CORRECT: Always scale for distance-based and gradient-based algorithms
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
sgd_clf.fit(X_train_scaled, y_train)
scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3)
print(f"Accuracy: {scores.mean():.3f}")  # Much better!
```

### Pitfall 4: Data Leakage in Cross-Validation
```python
# ❌ WRONG: Scaling before cross-validation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data including validation folds!
scores = cross_val_score(clf, X_scaled, y, cv=3)  # Results are too optimistic

# ✅ CORRECT: Use Pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SGDClassifier(random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=3)  # Scaling done separately for each fold
```

### Pitfall 5: Using predict() Instead of decision_function()
```python
# ❌ WRONG: Can't adjust threshold with predict()
y_pred = clf.predict(X_test)  # Binary predictions, threshold is fixed

# ✅ CORRECT: Use decision_function() or predict_proba()
# For SGDClassifier, SVC
y_scores = clf.decision_function(X_test)
y_pred = (y_scores > custom_threshold)  # Can adjust threshold

# For RandomForest, LogisticRegression
y_probas = clf.predict_proba(X_test)
y_pred = (y_probas[:, 1] > custom_threshold)  # Use positive class probability
```

### Pitfall 6: Not Handling Class Imbalance
```python
# ❌ WRONG: Ignoring imbalance
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_imbalanced)  # Majority class dominates

# ✅ CORRECT: Use class weights
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train_imbalanced)

# Or resample
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train_imbalanced)
clf.fit(X_resampled, y_resampled)
```

### Pitfall 7: Overfitting on Training Set
```python
# ❌ WRONG: Only checking training accuracy
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print(f"Training accuracy: {train_score:.3f}")  # 99.9% - too good to be true!

# ✅ CORRECT: Always use cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
# And keep test set for final evaluation
test_score = clf.score(X_test, y_test)
print(f"Test accuracy: {test_score:.3f}")
```

### Pitfall 8: Not Normalizing Confusion Matrix
```python
# ❌ WRONG: Raw counts can be misleading
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Hard to compare classes with different frequencies

# ✅ CORRECT: Normalize to see percentages
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    normalize="true",  # Normalize by row (recall per class)
    values_format=".2f"
)
plt.show()
```


## Quick Reference: When to Use What

### Classifier Selection Guide

| Scenario | Recommended Classifier | Why |
|----------|----------------------|-----|
| **Small dataset (<1000 samples)** | KNN, Decision Tree | Simple, no training needed (KNN) |
| **Medium dataset (1k-100k)** | Random Forest, SVM | Good performance, robust |
| **Large dataset (>100k)** | SGDClassifier | Efficient, online learning |
| **Need probability estimates** | Logistic Regression, Random Forest | Native probability support |
| **High-dimensional data** | Logistic Regression, Linear SVM | Works well with many features |
| **Non-linear decision boundary** | Random Forest, SVM (RBF kernel) | Can capture complex patterns |
| **Interpretability matters** | Logistic Regression, Decision Tree | Easy to explain |
| **Speed critical** | Logistic Regression, SGD | Fast training and prediction |

### Metric Selection Guide

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| **Balanced classes** | Accuracy, ROC AUC | Simple and intuitive |
| **Imbalanced classes** | F1 Score, PR AUC | Not misled by class imbalance |
| **False Positives costly** | Precision | Minimize false alarms |
| **False Negatives costly** | Recall | Don't miss positives |
| **Rare positive class** | PR Curve, F1 | ROC can be too optimistic |
| **Need single number** | F1 Score, ROC AUC | Easy to compare models |
| **Need to tune threshold** | PR Curve, ROC Curve | Shows trade-offs |

### Cross-Validation Strategy

| Data Size | CV Strategy | Folds |
|-----------|------------|-------|
| **< 1,000 samples** | Leave-One-Out or 10-fold | 10 or n |
| **1k - 10k samples** | Stratified K-Fold | 5-10 |
| **10k - 100k samples** | Stratified K-Fold | 3-5 |
| **> 100k samples** | Train-Val-Test split | N/A |
| **Time series data** | Time Series Split | 5 |
| **Highly imbalanced** | Stratified K-Fold | 5-10 |


## Final Checklist: Before Deploying Your Classifier

```
□ Created a test set and didn't touch it until the end
□ Checked for and handled class imbalance
□ Scaled features (if using SGD, SVM, KNN, or Logistic Regression)
□ Used appropriate metrics (not just accuracy)
□ Performed cross-validation (not just train-test split)
□ Analyzed confusion matrix and errors
□ Tuned decision threshold (for binary classification)
□ Tried multiple classifiers and compared them
□ Checked for overfitting (train vs validation performance)
□ Documented the model's limitations and edge cases
□ Tested on the test set only ONCE at the very end
□ Model performance is acceptable for the business problem
```


## Additional Resources

### Scikit-Learn Documentation
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Classifiers comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

### When Stuck
1. Start with a simple model (Logistic Regression or Random Forest)
2. Establish a baseline (even a dummy classifier)
3. Use cross-validation to measure progress
4. Analyze errors to understand what to improve
5. Try one thing at a time and measure impact

### Remember
> "Premature optimization is the root of all evil" - Donald Knuth

Start simple, measure carefully, and improve iteratively!

---

**End of Chapter 3 Notes**

*These notes are meant to be a practical companion. Refer back to them when working on classification problems, and don't hesitate to adapt the patterns to your specific use case.*
