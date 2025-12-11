"""
NASA TURBOFAN ENGINE RUL PREDICTION - 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*70)
print("NASA C-MAPSS TURBOFAN ENGINE - RUL PREDICTION PROJECT")
print("Dataset: FD001 (HPC Degradation, Sea Level)")
print("Using: SCIKIT-LEARN ONLY")
print("="*70)

#==============================================================================
# STEP 1: LOAD NASA C-MAPSS DATA
#==============================================================================
print("\n[STEP 1] Loading NASA C-MAPSS FD001 data...")

def load_cmapss_data(filepath):
    """Load NASA C-MAPSS dataset from text file"""
    columns = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]
    data = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    return data

# Load data
try:
    train_data = load_cmapss_data('train_FD001.txt')
    print(f"✓ Training data loaded: {len(train_data)} samples")
    print(f"✓ Number of engines: {train_data['unit'].nunique()}")
except FileNotFoundError:
    print("ERROR: train_FD001.txt not found!")
    print("Please place the file in the same directory as this script")
    exit()

try:
    test_data = load_cmapss_data('test_FD001.txt')
    print(f"✓ Test data loaded: {len(test_data)} samples")
    print(f"✓ Number of test engines: {test_data['unit'].nunique()}")
except FileNotFoundError:
    print("ERROR: test_FD001.txt not found!")
    exit()

try:
    test_rul = pd.read_csv('RUL_FD001.txt', header=None, names=['RUL'])
    print(f"✓ True RUL values loaded: {len(test_rul)} engines")
except FileNotFoundError:
    print("ERROR: RUL_FD001.txt not found!")
    exit()

#==============================================================================
# STEP 2: DATA PREPROCESSING
#==============================================================================
print("\n[STEP 2] Preprocessing data...")

def calculate_rul(data):
    """Calculate Remaining Useful Life for each cycle"""
    max_cycle = data.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    data = data.merge(max_cycle, on='unit', how='left')
    data['RUL'] = data['max_cycle'] - data['cycle']
    data.drop('max_cycle', axis=1, inplace=True)
    return data

def prepare_test_data(test_data, test_rul):
    """Prepare test data with true RUL values"""
    test_engines = []
    for unit_id in test_data['unit'].unique():
        unit_data = test_data[test_data['unit'] == unit_id].copy()
        max_cycle = unit_data['cycle'].max()
        true_rul = test_rul.iloc[unit_id - 1]['RUL']
        unit_data['RUL'] = true_rul + (max_cycle - unit_data['cycle'])
        test_engines.append(unit_data)
    return pd.concat(test_engines, ignore_index=True)

# Calculate RUL
train_data = calculate_rul(train_data)
test_data = prepare_test_data(test_data, test_rul)

print(f"  Training RUL range: {train_data['RUL'].min()} to {train_data['RUL'].max()} cycles")
print(f"  Test RUL range: {test_data['RUL'].min()} to {test_data['RUL'].max()} cycles")

# Feature columns
feature_cols = ['setting1', 'setting2', 'setting3'] + \
               [f'sensor{i}' for i in range(1, 22)]

# Extract multiple cycles per engine (better approach)
def get_features_multiple_cycles(data, max_rul=125):
    """
    Extract features from multiple cycles per engine.
    Clips RUL at max_rul to focus on recent degradation patterns.
    This is a common approach in the literature.
    """
    # Clip RUL at maximum value (common preprocessing)
    data_clipped = data.copy()
    data_clipped['RUL'] = data_clipped['RUL'].clip(upper=max_rul)
    
    # Sample multiple cycles per engine (every 5 cycles to reduce size)
    sampled_data = []
    for unit_id in data_clipped['unit'].unique():
        unit_data = data_clipped[data_clipped['unit'] == unit_id]
        # Take every 5th cycle
        sampled = unit_data.iloc[::5]
        sampled_data.append(sampled)
    
    combined = pd.concat(sampled_data, ignore_index=True)
    
    X = combined[feature_cols].values
    y_reg = combined['RUL'].values
    
    return X, y_reg

X_train, y_train_reg = get_features_multiple_cycles(train_data)
X_test, y_test_reg = get_features_multiple_cycles(test_data)

print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  RUL clipped at 125 cycles (common preprocessing)")

# Create 3-class classification targets
# Class 0: Healthy (RUL > 60 cycles)
# Class 1: Warning (30 < RUL ≤ 60 cycles)
# Class 2: Critical (RUL ≤ 30 cycles)
def create_three_classes(rul_values):
    """Convert RUL to 3 health classes"""
    classes = np.zeros(len(rul_values), dtype=int)
    classes[rul_values <= 30] = 2  # Critical
    classes[(rul_values > 30) & (rul_values <= 60)] = 1  # Warning
    classes[rul_values > 60] = 0  # Healthy
    return classes

y_train_class = create_three_classes(y_train_reg)
y_test_class = create_three_classes(y_test_reg)

print(f"  3-Class Classification:")
print(f"    Healthy (RUL > 60): Train={sum(y_train_class==0)}, Test={sum(y_test_class==0)}")
print(f"    Warning (30 < RUL ≤ 60): Train={sum(y_train_class==1)}, Test={sum(y_test_class==1)}")
print(f"    Critical (RUL ≤ 30): Train={sum(y_train_class==2)}, Test={sum(y_test_class==2)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#==============================================================================
# PART 1: REGRESSION METHODS
#==============================================================================
print("\n" + "="*70)
print("PART 1: REGRESSION METHODS (Predict Exact RUL)")
print("="*70)

#==============================================================================
# METHOD 1: LINEAR REGRESSION
#==============================================================================
print("\n[METHOD 1] LINEAR REGRESSION")
print("-" * 70)

start_time = time.time()

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_reg)

# Predict
y_pred_lr = lr.predict(X_test_scaled)
y_pred_lr = np.maximum(y_pred_lr, 0)  # RUL can't be negative

# Evaluate
rmse_lr = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
r2_lr = r2_score(y_test_reg, y_pred_lr)
time_lr = time.time() - start_time

print(f"Training Time: {time_lr:.2f} seconds")
print(f"RMSE: {rmse_lr:.2f} cycles")
print(f"MAE:  {mae_lr:.2f} cycles")
print(f"R²:   {r2_lr:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': np.abs(lr.coef_)
}).sort_values('coefficient', ascending=False)
print(f"\nTop 5 Important Features:")
print(feature_importance.head(5).to_string(index=False))

#==============================================================================
# METHOD 2: POLYNOMIAL REGRESSION
#==============================================================================
print("\n[METHOD 2] POLYNOMIAL REGRESSION (Degree 2)")
print("-" * 70)

start_time = time.time()

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"Features expanded: {X_train_scaled.shape[1]} → {X_train_poly.shape[1]}")

# Train Polynomial Regression
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train_reg)

# Predict
y_pred_poly = poly_lr.predict(X_test_poly)
y_pred_poly = np.maximum(y_pred_poly, 0)

# Evaluate
rmse_poly = np.sqrt(mean_squared_error(y_test_reg, y_pred_poly))
mae_poly = mean_absolute_error(y_test_reg, y_pred_poly)
r2_poly = r2_score(y_test_reg, y_pred_poly)
time_poly = time.time() - start_time

print(f"Training Time: {time_poly:.2f} seconds")
print(f"RMSE: {rmse_poly:.2f} cycles")
print(f"MAE:  {mae_poly:.2f} cycles")
print(f"R²:   {r2_poly:.4f}")

#==============================================================================
# PART 2: CLASSIFICATION METHODS
#==============================================================================
print("\n" + "="*70)
print("PART 2: CLASSIFICATION METHODS (Failure within 30 cycles)")
print("="*70)

#==============================================================================
# METHOD 3A: LOGISTIC REGRESSION (Multi-class)
#==============================================================================
print("\n[METHOD 3A] LOGISTIC REGRESSION (3 Classes)")
print("-" * 70)

start_time = time.time()

# Train Logistic Regression (multi-class)
log_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
log_reg.fit(X_train_scaled, y_train_class)

# Predict
y_pred_logreg = log_reg.predict(X_test_scaled)
y_proba_logreg = log_reg.predict_proba(X_test_scaled)

# Evaluate
acc_logreg = accuracy_score(y_test_class, y_pred_logreg)
prec_logreg = precision_score(y_test_class, y_pred_logreg, average='weighted', zero_division=0)
rec_logreg = recall_score(y_test_class, y_pred_logreg, average='weighted', zero_division=0)
f1_logreg = f1_score(y_test_class, y_pred_logreg, average='weighted', zero_division=0)
time_logreg = time.time() - start_time

print(f"Training Time: {time_logreg:.2f} seconds")
print(f"Accuracy:  {acc_logreg:.4f}")
print(f"Precision: {prec_logreg:.4f} (weighted)")
print(f"Recall:    {rec_logreg:.4f} (weighted)")
print(f"F1-Score:  {f1_logreg:.4f} (weighted)")

#==============================================================================
# METHOD 3B: SUPPORT VECTOR MACHINE (SVM) - Multi-class
#==============================================================================
print("\n[METHOD 3B] SUPPORT VECTOR MACHINE (SVM) - 3 Classes")
print("-" * 70)

start_time = time.time()

# Try both linear and RBF kernels
print("Testing Linear kernel...")
svm_linear = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_linear.fit(X_train_scaled, y_train_class)

y_pred_svm_linear = svm_linear.predict(X_test_scaled)
acc_svm_linear = accuracy_score(y_test_class, y_pred_svm_linear)

print(f"  Linear SVM - Accuracy: {acc_svm_linear:.4f}")

print("\nTesting RBF kernel with grid search...")
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.1]
}

svm_rbf_base = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf_grid = GridSearchCV(svm_rbf_base, param_grid, cv=3, 
                            scoring='accuracy', n_jobs=-1, verbose=0)
svm_rbf_grid.fit(X_train_scaled, y_train_class)

svm_rbf = svm_rbf_grid.best_estimator_
print(f"Best RBF parameters: {svm_rbf_grid.best_params_}")

y_pred_svm_rbf = svm_rbf.predict(X_test_scaled)

acc_svm_rbf = accuracy_score(y_test_class, y_pred_svm_rbf)
prec_svm_rbf = precision_score(y_test_class, y_pred_svm_rbf, average='weighted', zero_division=0)
rec_svm_rbf = recall_score(y_test_class, y_pred_svm_rbf, average='weighted', zero_division=0)
f1_svm_rbf = f1_score(y_test_class, y_pred_svm_rbf, average='weighted', zero_division=0)
time_svm = time.time() - start_time

print(f"\nBest SVM (RBF kernel):")
print(f"Training Time: {time_svm:.2f} seconds")
print(f"Accuracy:  {acc_svm_rbf:.4f}")
print(f"Precision: {prec_svm_rbf:.4f} (weighted)")
print(f"Recall:    {rec_svm_rbf:.4f} (weighted)")
print(f"F1-Score:  {f1_svm_rbf:.4f} (weighted)")

#==============================================================================
# METHOD 4: MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORKS
#==============================================================================
print("\n[METHOD 4] MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORKS")
print("-" * 70)

# MLP for REGRESSION
print("\nTraining MLP Regressor...")
start_time = time.time()

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=16,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)

mlp_reg.fit(X_train_scaled, y_train_reg)

y_pred_mlp_reg = mlp_reg.predict(X_test_scaled)
y_pred_mlp_reg = np.maximum(y_pred_mlp_reg, 0)

rmse_mlp = np.sqrt(mean_squared_error(y_test_reg, y_pred_mlp_reg))
mae_mlp = mean_absolute_error(y_test_reg, y_pred_mlp_reg)
r2_mlp = r2_score(y_test_reg, y_pred_mlp_reg)
time_mlp_reg = time.time() - start_time

print(f"Training Time: {time_mlp_reg:.2f} seconds")
print(f"Iterations: {mlp_reg.n_iter_}")
print(f"RMSE: {rmse_mlp:.2f} cycles")
print(f"MAE:  {mae_mlp:.2f} cycles")
print(f"R²:   {r2_mlp:.4f}")

# MLP for CLASSIFICATION
print("\nTraining MLP Classifier...")
start_time = time.time()

mlp_class = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=16,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)

mlp_class.fit(X_train_scaled, y_train_class)

y_pred_mlp_class = mlp_class.predict(X_test_scaled)
y_proba_mlp = mlp_class.predict_proba(X_test_scaled)[:, 1]

acc_mlp = accuracy_score(y_test_class, y_pred_mlp_class)
prec_mlp = precision_score(y_test_class, y_pred_mlp_class, average='weighted', zero_division=0)
rec_mlp = recall_score(y_test_class, y_pred_mlp_class, average='weighted', zero_division=0)
f1_mlp = f1_score(y_test_class, y_pred_mlp_class, average='weighted', zero_division=0)
time_mlp_class = time.time() - start_time

print(f"Training Time: {time_mlp_class:.2f} seconds")
print(f"Iterations: {mlp_class.n_iter_}")
print(f"Accuracy:  {acc_mlp:.4f}")
print(f"Precision: {prec_mlp:.4f} (weighted)")
print(f"Recall:    {rec_mlp:.4f} (weighted)")
print(f"F1-Score:  {f1_mlp:.4f} (weighted)")

#==============================================================================
# RESULTS COMPARISON
#==============================================================================
print("\n" + "="*70)
print("FINAL RESULTS COMPARISON")
print("="*70)

# Regression Results
print("\n--- REGRESSION RESULTS (Predict RUL in cycles) ---")
regression_results = pd.DataFrame({
    'Method': ['Linear Regression', 'Polynomial Regression', 'MLP Neural Network'],
    'RMSE': [rmse_lr, rmse_poly, rmse_mlp],
    'MAE': [mae_lr, mae_poly, mae_mlp],
    'R²': [r2_lr, r2_poly, r2_mlp],
    'Time (s)': [time_lr, time_poly, time_mlp_reg]
})
print(regression_results.to_string(index=False))

# Classification Results
print("\n--- CLASSIFICATION RESULTS (3 Classes: Healthy/Warning/Critical) ---")
classification_results = pd.DataFrame({
    'Method': ['Logistic Regression', 'SVM (RBF)', 'MLP Neural Network'],
    'Accuracy': [acc_logreg, acc_svm_rbf, acc_mlp],
    'Precision': [prec_logreg, prec_svm_rbf, prec_mlp],
    'Recall': [rec_logreg, rec_svm_rbf, rec_mlp],
    'F1-Score': [f1_logreg, f1_svm_rbf, f1_mlp],
    'Time (s)': [time_logreg, time_svm, time_mlp_class]
})
print(classification_results.to_string(index=False))

#==============================================================================
# VISUALIZATIONS
#==============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 12))

# ===== REGRESSION PLOTS =====
ax1 = plt.subplot(3, 3, 1)
plt.scatter(y_test_reg, y_pred_lr, alpha=0.6, s=30)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual RUL (cycles)', fontsize=10)
plt.ylabel('Predicted RUL (cycles)', fontsize=10)
plt.title(f'Linear Regression\nRMSE={rmse_lr:.2f}', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
plt.scatter(y_test_reg, y_pred_poly, alpha=0.6, s=30, color='orange')
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual RUL (cycles)', fontsize=10)
plt.ylabel('Predicted RUL (cycles)', fontsize=10)
plt.title(f'Polynomial Regression\nRMSE={rmse_poly:.2f}', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
plt.scatter(y_test_reg, y_pred_mlp_reg, alpha=0.6, s=30, color='green')
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual RUL (cycles)', fontsize=10)
plt.ylabel('Predicted RUL (cycles)', fontsize=10)
plt.title(f'MLP Neural Network\nRMSE={rmse_mlp:.2f}', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 3, 4)
methods = ['Linear', 'Polynomial', 'MLP']
rmse_values = [rmse_lr, rmse_poly, rmse_mlp]
colors = ['steelblue', 'orange', 'green']
bars = plt.bar(methods, rmse_values, color=colors, alpha=0.7)
plt.ylabel('RMSE (cycles)', fontsize=10)
plt.title('Regression Performance', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# ===== CLASSIFICATION PLOTS =====
class_labels = ['Healthy', 'Warning', 'Critical']

ax5 = plt.subplot(3, 3, 5)
cm_logreg = confusion_matrix(y_test_class, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Actual', fontsize=10)
plt.title(f'Logistic Regression\nAcc={acc_logreg:.3f}', fontsize=11, fontweight='bold')

ax6 = plt.subplot(3, 3, 6)
cm_svm = confusion_matrix(y_test_class, y_pred_svm_rbf)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Actual', fontsize=10)
plt.title(f'SVM (RBF)\nAcc={acc_svm_rbf:.3f}', fontsize=11, fontweight='bold')

ax7 = plt.subplot(3, 3, 7)
cm_mlp = confusion_matrix(y_test_class, y_pred_mlp_class)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Actual', fontsize=10)
plt.title(f'MLP Neural Network\nAcc={acc_mlp:.3f}', fontsize=11, fontweight='bold')

# Class distribution plot
ax8 = plt.subplot(3, 3, 8)
class_names = ['Healthy\n(>60)', 'Warning\n(30-60)', 'Critical\n(≤30)']
train_counts = [sum(y_train_class==0), sum(y_train_class==1), sum(y_train_class==2)]
test_counts = [sum(y_test_class==0), sum(y_test_class==1), sum(y_test_class==2)]

x = np.arange(len(class_names))
width = 0.35

plt.bar(x - width/2, train_counts, width, label='Train', color='steelblue', alpha=0.7)
plt.bar(x + width/2, test_counts, width, label='Test', color='orange', alpha=0.7)

plt.xlabel('Health Class', fontsize=10)
plt.ylabel('Number of Samples', fontsize=10)
plt.title('Class Distribution', fontsize=11, fontweight='bold')
plt.xticks(x, class_names, fontsize=9)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3, axis='y')

ax9 = plt.subplot(3, 3, 9)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
logreg_scores = [acc_logreg, prec_logreg, rec_logreg, f1_logreg]
svm_scores = [acc_svm_rbf, prec_svm_rbf, rec_svm_rbf, f1_svm_rbf]
mlp_scores = [acc_mlp, prec_mlp, rec_mlp, f1_mlp]

x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, logreg_scores, width, label='Logistic Reg', color='steelblue', alpha=0.7)
plt.bar(x, svm_scores, width, label='SVM', color='orange', alpha=0.7)
plt.bar(x + width, mlp_scores, width, label='MLP', color='green', alpha=0.7)

plt.xlabel('Metrics', fontsize=10)
plt.ylabel('Score', fontsize=10)
plt.title('Classification Metrics Comparison', fontsize=11, fontweight='bold')
plt.xticks(x, metrics, fontsize=9, rotation=15)
plt.ylim([0, 1.1])
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('nasa_turbofan_results_final.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: nasa_turbofan_results_final.png")

# Save results
regression_results.to_csv('regression_results.csv', index=False)
classification_results.to_csv('classification_results.csv', index=False)
print("✓ Results saved to CSV files")

#==============================================================================
# SUMMARY
#==============================================================================
print("\n" + "="*70)
print("PROJECT COMPLETE!")
print("="*70)

print("\n METHODS DEMONSTRATED (ALL SCIKIT-LEARN):")
print("   REGRESSION (2 methods):")
print("   1. ✓ Linear Regression")
print("   2. ✓ Polynomial Regression (degree 2)")
print("\n   CLASSIFICATION (3 methods for 3 classes):")
print("   3. ✓ Logistic Regression")
print("   4. ✓ Support Vector Machine (SVM)")
print("   5. ✓ Multi-Layer Perceptron (MLP)")
print("\n   Classes: Healthy (RUL>60) | Warning (30<RUL≤60) | Critical (RUL≤30)")

print("\n KEY FINDINGS:")
best_reg_idx = regression_results['RMSE'].idxmin()
print(f"   • Best Regression: {regression_results.loc[best_reg_idx, 'Method']}")
print(f"     RMSE = {regression_results.loc[best_reg_idx, 'RMSE']:.2f} cycles")

best_class_idx = classification_results['Accuracy'].idxmax()
print(f"   • Best Classification: {classification_results.loc[best_class_idx, 'Method']}")
print(f"     Accuracy = {classification_results.loc[best_class_idx, 'Accuracy']:.4f}")

print("\n OUTPUTS:")
print("   • nasa_turbofan_results_final.png")
print("   • regression_results.csv")
print("   • classification_results.csv")

print("\n READY FOR YOUR REPORT!")
print("="*70)

plt.show()