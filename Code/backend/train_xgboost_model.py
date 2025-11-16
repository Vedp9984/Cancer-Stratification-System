#!/usr/bin/env python3
"""
Train XGBoost Model for Cancer Risk Stratification

This script trains an XGBoost classifier on sample radiology reports
to enable ML-based risk prediction in the main risk_model.py script.

Usage: python3 train_xgboost_model.py
"""

import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from risk_model import (
    initialize_biobert,
    clean_text,
    chexpert_label_extractor,
    calculate_chexpert_score,
    biobert_analyze,
    calculate_biobert_score,
    extract_clinical_features,
    calculate_clinical_features_score,
    create_unified_feature_vector,
    FEATURE_COLUMNS
)

# Sample training data (from notebook)
SAMPLE_REPORTS = [
    {
        'report_text': """
        CHEST X-RAY, PA AND LATERAL
        CLINICAL HISTORY: 65-year-old male with acute respiratory distress
        FINDINGS:
        Large bilateral pleural effusions with associated atelectasis. 
        Extensive consolidation in bilateral lower lobes consistent with pneumonia.
        Mild cardiomegaly noted. No pneumothorax identified.
        Multiple pulmonary opacities throughout both lung fields.
        IMPRESSION:
        1. Severe bilateral pneumonia with large pleural effusions
        2. Moderate cardiomegaly
        3. Extensive lung opacities suggesting acute respiratory infection
        RECOMMENDATION: Immediate clinical correlation and ICU monitoring recommended.
        """,
        'expected_risk': 'HIGH',
        'patient_age': 65
    },
    {
        'report_text': """
        CHEST X-RAY, FRONTAL VIEW
        CLINICAL HISTORY: 72-year-old female with suspected pneumothorax
        FINDINGS:
        Large right-sided pneumothorax with significant lung collapse, measuring 
        approximately 40% of the hemithorax. Trachea shifted slightly to the left.
        No pleural effusion. Chest tube placement recommended.
        Suspected rib fracture at the 7th rib on the right.
        IMPRESSION:
        1. Large right pneumothorax (40%) with mediastinal shift
        2. Right 7th rib fracture
        3. Urgent intervention required
        RECOMMENDATION: Immediate chest tube placement advised.
        """,
        'expected_risk': 'HIGH',
        'patient_age': 72
    },
    {
        'report_text': """
        CHEST RADIOGRAPH
        CLINICAL HISTORY: 45-year-old male, routine screening
        FINDINGS:
        Suspicious mass lesion in the right upper lobe measuring approximately 
        2.5 cm in diameter. Borders are irregular and spiculated. 
        No evidence of pleural effusion or pneumothorax.
        Hilar lymphadenopathy present bilaterally.
        Heart size is normal.
        IMPRESSION:
        1. Right upper lobe mass, concerning for malignancy
        2. Bilateral hilar lymphadenopathy
        3. Further evaluation with CT chest recommended
        RECOMMENDATION: CT chest with contrast and possible biopsy.
        """,
        'expected_risk': 'HIGH',
        'patient_age': 45
    },
    {
        'report_text': """
        CHEST X-RAY
        CLINICAL HISTORY: 55-year-old female with mild cough
        FINDINGS:
        Patchy opacity in the left lower lobe suggesting mild infiltrate.
        Small amount of atelectasis at the left base.
        No pleural effusion or pneumothorax.
        Cardiac silhouette is within normal limits.
        No focal consolidation.
        IMPRESSION:
        1. Mild left lower lobe infiltrate, possibly early pneumonia
        2. Minor atelectasis
        RECOMMENDATION: Follow-up chest X-ray in 2 weeks if symptoms persist.
        """,
        'expected_risk': 'MEDIUM',
        'patient_age': 55
    },
    {
        'report_text': """
        CHEST RADIOGRAPH, PA VIEW
        CLINICAL HISTORY: 38-year-old male, post-operative
        FINDINGS:
        Small right pleural effusion, likely post-surgical.
        Mild pulmonary edema bilaterally.
        Central venous catheter in appropriate position.
        No pneumothorax. Cardiac size is mildly enlarged.
        Support devices including monitoring leads visualized.
        IMPRESSION:
        1. Small right pleural effusion (post-operative)
        2. Mild pulmonary edema
        3. Cardiomegaly, mild
        RECOMMENDATION: Continue monitoring. Repeat imaging in 48 hours.
        """,
        'expected_risk': 'MEDIUM',
        'patient_age': 38
    },
    {
        'report_text': """
        CHEST X-RAY
        CLINICAL HISTORY: 50-year-old female, follow-up
        FINDINGS:
        Small nodule noted in right mid lung field, measuring 8mm.
        No infiltrates or consolidation.
        Lungs are otherwise clear bilaterally.
        No pleural effusion or pneumothorax.
        Cardiac and mediastinal contours are normal.
        IMPRESSION:
        1. Small right lung nodule (8mm), indeterminate
        2. Otherwise unremarkable chest X-ray
        RECOMMENDATION: CT chest for nodule characterization recommended.
        """,
        'expected_risk': 'MEDIUM',
        'patient_age': 50
    },
    {
        'report_text': """
        CHEST RADIOGRAPH
        CLINICAL HISTORY: 30-year-old male, pre-employment physical
        FINDINGS:
        The lungs are clear bilaterally with no focal consolidation, 
        infiltrate, or mass lesion. No pleural effusion or pneumothorax.
        Cardiac silhouette is normal in size and contour.
        Mediastinum is within normal limits.
        Bony structures are intact.
        IMPRESSION:
        1. Normal chest radiograph
        2. No acute cardiopulmonary disease
        RECOMMENDATION: None. Routine care.
        """,
        'expected_risk': 'LOW',
        'patient_age': 30
    },
    {
        'report_text': """
        CHEST X-RAY, FRONTAL AND LATERAL
        CLINICAL HISTORY: 42-year-old female, annual checkup
        FINDINGS:
        Clear lung fields bilaterally. No infiltrates, masses, or nodules.
        Cardiac size and mediastinal contours are unremarkable.
        No pleural effusion. No pneumothorax.
        Skeletal structures show no acute abnormality.
        Soft tissues are unremarkable.
        IMPRESSION:
        1. No acute findings
        2. Normal cardiopulmonary examination
        RECOMMENDATION: Continue routine health maintenance.
        """,
        'expected_risk': 'LOW',
        'patient_age': 42
    }
]

def process_training_data(biobert_model, biobert_tokenizer):
    """Process all training reports to extract features"""
    
    print("\n" + "="*80)
    print("PROCESSING TRAINING DATA")
    print("="*80)
    
    feature_vectors = []
    labels = []
    
    for i, report in enumerate(SAMPLE_REPORTS, 1):
        print(f"\nProcessing report {i}/{len(SAMPLE_REPORTS)} ({report['expected_risk']})...")
        
        # Clean text
        cleaned_text = clean_text(report['report_text'])
        
        # CheXpert analysis
        chexpert_findings = chexpert_label_extractor(cleaned_text)
        chexpert_score = calculate_chexpert_score(chexpert_findings)
        
        # BioBERT analysis
        biobert_result = biobert_analyze(cleaned_text, biobert_model, biobert_tokenizer)
        biobert_score = calculate_biobert_score(cleaned_text, biobert_result)
        
        # Clinical features
        clinical_features = extract_clinical_features(cleaned_text, report['patient_age'])
        clinical_score = calculate_clinical_features_score(clinical_features)
        
        # Create unified feature vector
        feature_vector = create_unified_feature_vector(
            chexpert_findings, chexpert_score,
            biobert_result, biobert_score,
            clinical_features, clinical_score
        )
        
        # Convert to array
        feature_array = [feature_vector[col] for col in FEATURE_COLUMNS]
        feature_vectors.append(feature_array)
        labels.append(report['expected_risk'])
        
        print(f"  ✓ Features extracted: {len(feature_array)} dimensions")
    
    return np.array(feature_vectors), labels

def train_model(X, y):
    """Train XGBoost model and scaler"""
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Normalize features
    print("\nNormalizing features with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    print(f"✓ Features normalized: shape {X_normalized.shape}")
    
    # Encode labels
    label_mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    y_numeric = np.array([label_mapping[label] for label in y])
    
    print(f"\nLabel distribution:")
    for label, code in label_mapping.items():
        count = np.sum(y_numeric == code)
        print(f"  {label:10s} ({code}): {count} samples")
    
    # Configure XGBoost
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    print(f"\nTraining XGBoost classifier...")
    print(f"  Parameters: {xgb_params}")
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_normalized, y_numeric)
    
    print(f"✓ Model trained successfully")
    
    # Evaluate on training data (for verification)
    predictions = xgb_model.predict(X_normalized)
    accuracy = np.mean(predictions == y_numeric)
    print(f"\nTraining accuracy: {accuracy*100:.1f}%")
    
    # Show predictions
    print("\nPrediction verification:")
    for i, (pred, true) in enumerate(zip(predictions, y_numeric)):
        pred_label = list(label_mapping.keys())[pred]
        true_label = list(label_mapping.keys())[true]
        match = "✓" if pred == true else "✗"
        print(f"  Report {i+1}: Predicted={pred_label:6s}, Actual={true_label:6s} {match}")
    
    return xgb_model, scaler

def save_model(xgb_model, scaler):
    """Save trained model and scaler"""
    
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    # Save XGBoost model
    model_path = 'xgboost_risk_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"✓ XGBoost model saved to: {model_path}")
    
    # Save scaler
    scaler_path = 'feature_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Feature scaler saved to: {scaler_path}")
    
    # Display feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    importance_values = xgb_model.feature_importances_
    feature_importance = [
        (FEATURE_COLUMNS[i], importance_values[i]) 
        for i in range(len(FEATURE_COLUMNS))
    ]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    print("-"*80)
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        bar_length = int(importance * 50)
        bar = "█" * bar_length
        print(f"{i:2d}. {feature:35s} {importance:.4f} {bar}")
    
    print("\n" + "="*80)

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("XGBOOST MODEL TRAINING SCRIPT")
    print("="*80)
    print(f"\nThis script will train an XGBoost model on {len(SAMPLE_REPORTS)} sample reports")
    print("The trained model will be saved and used by risk_model.py for predictions")
    
    # Initialize BioBERT
    print("\n" + "="*80)
    print("INITIALIZING BIOBERT")
    print("="*80)
    biobert_model, biobert_tokenizer = initialize_biobert()
    
    if biobert_model is None or biobert_tokenizer is None:
        print("Error: Failed to initialize BioBERT model")
        sys.exit(1)
    
    # Process training data
    X, y = process_training_data(biobert_model, biobert_tokenizer)
    
    # Train model
    xgb_model, scaler = train_model(X, y)
    
    # Save model
    save_model(xgb_model, scaler)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print("\nThe following files have been created:")
    print("  1. xgboost_risk_model.pkl  - Trained XGBoost classifier")
    print("  2. feature_scaler.pkl      - Feature normalization scaler")
    print("\nYou can now use risk_model.py with full XGBoost predictions!")
    print("\nUsage:")
    print("  python3 risk_model.py <path_to_report_image>")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
