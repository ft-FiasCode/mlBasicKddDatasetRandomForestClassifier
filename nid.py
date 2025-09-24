import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load data from CSV file into a pandas DataFrame
# DataFrame: 2D table-like data structure with rows (samples) and columns (features & label)
data = pd.read_csv("KDDDataset.txt")
df = pd.DataFrame(data)
print(df.head())  # Print first 5 rows to inspect data

# Separate features and label columns from DataFrame
# Features: Input variables used by the model to make predictions (independent variables)
# Label (target): The variable to predict (dependent variable, e.g. "normal" or "attack")
x = df.iloc[:, :-1].copy()   # All columns except the last are features
y = df.iloc[:, -1].copy()    # Last column is the label

# Handle missing values in features by replacing with the most frequent value per column
imputer = SimpleImputer(strategy='most_frequent')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# Identify categorical columns with text data needing conversion to numbers
cat_cols = [col for col in x.columns if x[col].dtype == 'object']

# Encode categorical features as numeric codes for model compatibility
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()  # Converts text labels to integer codes
    x[col] = le.fit_transform(x[col].astype(str))
    label_encoders[col] = le  # Save encoders if needed for decoding later

# Convert target labels to binary numeric form
# 0 if label equals “normal”, 1 otherwise (indicating some kind of attack)
y = y.apply(lambda v: 0 if str(v).lower() == "normal" else 1)

"""
STRATIFY use:
Keeps balance of normal/attack ratio same in both sets (train and test).
If you split randomly without stratify, you might accidentally put mostly normal 
samples in training and very few attacks in testing.
"""

# Split dataset into training and testing sets
# train_test_split: function to split data
# train_size=0.7 means 70% data for training, 30% for testing
# stratify=y maintains the original class distribution in train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, stratify=y
)

# Define hyperparameter grid for Random Forest model tuning
param_grid = {
    'n_estimators': [50, 100],          # Number of trees in the forest
    'max_depth': [None, 10, 20],       # Maximum depth of each tree
    'min_samples_split': [2, 5]        # Minimum samples needed to split a node
}

"""
1.classify → variable name for classifier model
2.RandomForestClassifier → ML algorithm that builds many decision trees and combines them to improve accuracy
3.n_estimators=100 → use 100 trees in the forest
4.random_state=42 → same randomness every run for reproducibility
"""

# RandomForestClassifier: Ensemble of decision trees for classification tasks
rfc = RandomForestClassifier(random_state=42)

# GridSearchCV performs cross-validation and searches the best hyperparameters from param_grid
# cv=3 means 3-fold cross-validation
# scoring='f1' optimizes the F1 score metric (harmonic mean of precision and recall)
grid_search = GridSearchCV(rfc, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(x_train, y_train)  # Train models with different parameters and select the best

best_model = grid_search.best_estimator_  # Extract best model after tuning

print(f"Best hyperparameters: {grid_search.best_params_}")

# Make predictions on the test dataset using the trained best model
prediction = best_model.predict(x_test)

# confusion_matrix: Shows number of correct/incorrect predictions split by classes
# Labels: [0, 1] represent Normal (0) and Attack (1) classes
cm = confusion_matrix(y_test, prediction, labels=[0, 1])
print("\nConfusion Matrix:")
print(pd.DataFrame(cm,
                   index=["Actual Normal (0)", "Actual Attack (1)"],
                   columns=["Predicted Normal (0)", "Predicted Attack (1)"]))


"""
Accuracy → overall correctness
Precision → reliability of “attack” alarms
Recall → ability to catch actual attacks
F1-Score → tradeoff between precision & recall
"""

# classification_report: Summary of key classification metrics per class
print("\nClassification Report:")
print(classification_report(y_test, prediction, labels=[0, 1],
                            target_names=["Normal (0)", "Attack (1)"],
                            zero_division=0))  # zero_division=0 avoids divide-by-zero errors
# End of code