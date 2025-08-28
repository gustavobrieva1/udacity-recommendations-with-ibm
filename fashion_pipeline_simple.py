#!/usr/bin/env python3
"""
Fashion Forward Forecasting: Simplified ML Pipeline
StyleSense Product Recommendation Prediction

This is a streamlined version focusing on core requirements.
"""

import pandas as pd
import numpy as np
import re

# ML Pipeline imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

print("=== Fashion Forward Forecasting - Simplified Pipeline ===")
print("Loading StyleSense customer review data...")

# Load data
df = pd.read_csv('data/reviews.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:")
print(df['Recommended IND'].value_counts(normalize=True))

# Data exploration
print(f"\nNumerical features statistics:")
numerical_features = ['Age', 'Positive Feedback Count']
print(df[numerical_features].describe())

print(f"\nCategorical features unique values:")
categorical_features = ['Division Name', 'Department Name', 'Class Name']
for feature in categorical_features:
    print(f"{feature}: {df[feature].nunique()} unique values")

print(f"\nText features length statistics:")
text_features = ['Title', 'Review Text']
for feature in text_features:
    lengths = df[feature].str.len()
    print(f"{feature} - Mean: {lengths.mean():.1f}, Max: {lengths.max()}")

# Prepare features and target
X = df.drop('Recommended IND', axis=1)
y = df['Recommended IND'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, random_state=27
)

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")

def simple_text_cleaner(text):
    """Simple text cleaning function."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean text data
print("\n=== Preprocessing Text Data ===")
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

X_train_clean['Title'] = X_train_clean['Title'].apply(simple_text_cleaner)
X_train_clean['Review Text'] = X_train_clean['Review Text'].apply(simple_text_cleaner)
X_test_clean['Title'] = X_test_clean['Title'].apply(simple_text_cleaner)
X_test_clean['Review Text'] = X_test_clean['Review Text'].apply(simple_text_cleaner)

print("Text preprocessing completed.")

def create_pipeline():
    """Create comprehensive ML pipeline."""
    
    # Numerical preprocessing
    numerical_features = ['Age', 'Positive Feedback Count']
    numerical_transformer = StandardScaler()
    
    # Categorical preprocessing  
    categorical_features = ['Division Name', 'Department Name', 'Class Name']
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Clothing ID as categorical
    clothing_id_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Text preprocessing
    title_vectorizer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    review_vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Combine all preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features),
            ('clothing_id', clothing_id_transformer, ['Clothing ID']),
            ('title_tfidf', title_vectorizer, 'Title'),
            ('review_tfidf', review_vectorizer, 'Review Text')
        ],
        remainder='drop'
    )
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=27, max_iter=1000))
    ])
    
    return pipeline

print("\n=== Building and Training Pipeline ===")
pipeline = create_pipeline()

print("Training initial pipeline...")
pipeline.fit(X_train_clean, y_train)

# Make predictions
y_train_pred = pipeline.predict(X_train_clean)
y_test_pred = pipeline.predict(X_test_clean)

print("\n=== Initial Model Performance ===")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Test Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_test_pred):.4f}")

print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(pipeline, X_train_clean, y_train, cv=5, scoring='f1')
print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n=== Hyperparameter Tuning ===")
param_grid = [
    {
        'classifier': [LogisticRegression(random_state=27, max_iter=2000)],
        'classifier__C': [0.1, 1.0, 10.0]
    },
    {
        'classifier': [RandomForestClassifier(random_state=27, n_jobs=-1)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20]
    }
]

base_pipeline = create_pipeline()
grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Performing grid search...")
grid_search.fit(X_train_clean, y_train)

print(f"\n=== Best Parameters ===")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

# Final evaluation
best_model = grid_search.best_estimator_
y_test_pred_best = best_model.predict(X_test_clean)

print(f"\n=== Final Model Performance ===")
print(f"Best model: {type(best_model.named_steps['classifier']).__name__}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_best):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred_best):.4f}")
print(f"Test Recall: {recall_score(y_test, y_test_pred_best):.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_test_pred_best):.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred_best))

print("\n=== Sample Predictions ===")
# Show sample predictions
for i in range(3):
    actual = y_test.iloc[i]
    predicted = y_test_pred_best[i]
    proba = best_model.predict_proba(X_test_clean.iloc[[i]])[0]
    
    print(f"\nSample {i+1}:")
    print(f"Title: '{X_test.iloc[i]['Title'][:50]}...'")
    print(f"Age: {X_test.iloc[i]['Age']}, Dept: {X_test.iloc[i]['Department Name']}")
    print(f"Actual: {'✅ Recommended' if actual == 1 else '❌ Not Recommended'}")
    print(f"Predicted: {'✅ Recommended' if predicted == 1 else '❌ Not Recommended'}")
    print(f"Confidence: {proba[1]:.3f}")

print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE - SIMPLIFIED VERSION!")
print("="*60)
print("✅ Successfully created ML pipeline with:")
print("   • Mixed data type handling (numerical, categorical, text)")
print("   • Proper preprocessing for each data type")
print("   • TF-IDF vectorization for text features")
print("   • Hyperparameter tuning with GridSearchCV")
print("   • Cross-validation evaluation")
print("   • Comprehensive performance metrics")
print("\n🏆 Ready for StyleSense deployment!")
print("="*60)