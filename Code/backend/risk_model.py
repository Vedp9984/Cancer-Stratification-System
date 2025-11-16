#!/usr/bin/env python3
"""
Cancer Risk Stratification Model
Analyzes radiology reports using OCR and NLP to generate risk scores.

Usage: python3 risk_model.py <path_to_report_image>
Output: CSV file with risk score and medical condition summary
"""

import sys
import os
import re
import warnings
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModel
import torch
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration and Constants
# ============================================================================

BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CheXpert disease labels and keywords
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

CHEXPERT_KEYWORDS = {
    'No Finding': ['normal', 'clear', 'no acute', 'unremarkable'],
    'Enlarged Cardiomediastinum': ['enlarged cardiomediastinum', 'widened mediastinum'],
    'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
    'Lung Opacity': ['opacity', 'opacities', 'infiltrate', 'infiltrates'],
    'Lung Lesion': ['lesion', 'mass', 'nodule', 'nodules'],
    'Edema': ['edema', 'pulmonary edema', 'fluid overload'],
    'Consolidation': ['consolidation', 'consolidated'],
    'Pneumonia': ['pneumonia', 'pneumonitis', 'infection'],
    'Atelectasis': ['atelectasis', 'collapse', 'volume loss'],
    'Pneumothorax': ['pneumothorax', 'collapsed lung', 'air in pleural'],
    'Pleural Effusion': ['pleural effusion', 'effusion', 'fluid'],
    'Pleural Other': ['pleural thickening', 'pleural'],
    'Fracture': ['fracture', 'broken', 'rib fracture'],
    'Support Devices': ['tube', 'catheter', 'line', 'device', 'pacemaker']
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'biobert': 0.40,
    'chexpert': 0.30,
    'xgboost': 0.20,
    'clinical': 0.10
}

# Risk classification thresholds
RISK_THRESHOLDS = {
    'LOW': (0, 30),
    'MEDIUM': (30, 70),
    'HIGH': (70, 100)
}

# Feature columns for unified feature vector (14 features)
FEATURE_COLUMNS = [
    'chexpert_score',
    'chexpert_positive_findings',
    'chexpert_high_risk_present',
    'biobert_score',
    'biobert_embedding_mean',
    'biobert_embedding_std',
    'clinical_score',
    'clinical_bilateral',
    'clinical_severe',
    'clinical_acute',
    'clinical_pathology_count',
    'clinical_high_severity',
    'clinical_negative_indicators',
    'clinical_age_risk'
]

# ============================================================================
# Model Initialization
# ============================================================================

def initialize_biobert():
    """Initialize BioBERT model and tokenizer"""
    print("Loading BioBERT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
        model = model.to(DEVICE)
        model.eval()
        print(f"✓ BioBERT loaded on {DEVICE}")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Error loading BioBERT: {e}")
        return None, None

def initialize_xgboost_model():
    """Initialize or train XGBoost model"""
    model_path = 'xgboost_risk_model.pkl'
    scaler_path = 'feature_scaler.pkl'
    
    # Try to load pre-trained model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, 'rb') as f:
                xgb_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded pre-trained XGBoost model from {model_path}")
            return xgb_model, scaler
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("  Will train new model on first use")
    
    # Return None if model doesn't exist - will be trained on first batch
    return None, None

def train_xgboost_model(feature_matrix, labels):
    """
    Train XGBoost model on feature matrix
    This is called when processing the first report or training data
    """
    print("\nTraining XGBoost model...")
    
    # Encode labels: LOW=0, MEDIUM=1, HIGH=2
    label_mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    y_numeric = np.array([label_mapping.get(label, 1) for label in labels])
    
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
    
    # Train model
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(feature_matrix, y_numeric)
    
    print("✓ XGBoost model trained")
    
    # Save model
    try:
        with open('xgboost_risk_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        print("✓ Model saved to xgboost_risk_model.pkl")
    except Exception as e:
        print(f"⚠ Could not save model: {e}")
    
    return xgb_model

# ============================================================================
# OCR and Text Processing
# ============================================================================

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    print(f"Extracting text from image: {image_path}")
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print(f"✓ Extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"✗ OCR extraction failed: {e}")
        sys.exit(1)

def clean_text(text):
    """Clean and normalize extracted text"""
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    # Normalize line breaks
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    # Normalize multiple spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Remove extra newlines
    cleaned = re.sub(r'\n+', ' ', cleaned)
    return cleaned

# ============================================================================
# CheXpert Labeler
# ============================================================================

def chexpert_label_extractor(text):
    """Extract CheXpert labels from text"""
    text_lower = text.lower()
    findings = {}
    
    for label, keywords in CHEXPERT_KEYWORDS.items():
        match_count = sum(1 for keyword in keywords if keyword in text_lower)
        if match_count > 0:
            confidence = min(0.5 + (match_count * 0.3), 1.0)
            findings[label] = round(confidence, 3)
        else:
            findings[label] = 0.0
    
    return findings

def calculate_chexpert_score(findings):
    """Calculate CheXpert risk score"""
    severity_weights = {
        'No Finding': -0.8,
        'Enlarged Cardiomediastinum': 0.6,
        'Cardiomegaly': 0.5,
        'Lung Opacity': 0.6,
        'Lung Lesion': 0.9,
        'Edema': 0.7,
        'Consolidation': 0.8,
        'Pneumonia': 0.9,
        'Atelectasis': 0.5,
        'Pneumothorax': 0.95,
        'Pleural Effusion': 0.7,
        'Pleural Other': 0.4,
        'Fracture': 0.6,
        'Support Devices': 0.3
    }
    
    total_score = 0
    positive_findings = 0
    
    for label, confidence in findings.items():
        if confidence > 0 and label != 'No Finding':
            weight = severity_weights.get(label, 0.5)
            total_score += confidence * weight
            positive_findings += 1
    
    if findings.get('No Finding', 0) > 0.5:
        total_score = max(0, total_score - 0.5)
    
    if positive_findings > 0:
        normalized_score = min(total_score / (positive_findings * 0.7), 1.0)
    else:
        normalized_score = 0.1
    
    return normalized_score

# ============================================================================
# BioBERT Analysis
# ============================================================================

def biobert_analyze(text, model, tokenizer):
    """Analyze text using BioBERT"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pooled_output = outputs.pooler_output
        
        embedding_vector = pooled_output.cpu().numpy().flatten()
        
        return {
            'embedding_mean': float(np.mean(np.abs(embedding_vector))),
            'embedding_std': float(np.std(embedding_vector)),
            'embedding_max': float(np.max(np.abs(embedding_vector)))
        }
    except Exception as e:
        print(f"✗ BioBERT analysis error: {e}")
        return None

def calculate_biobert_score(text, analysis_result):
    """Calculate BioBERT risk score"""
    if analysis_result is None:
        return 0.5
    
    high_severity_terms = [
        'severe', 'extensive', 'large', 'significant', 'massive', 'acute',
        'critical', 'urgent', 'emergency', 'immediate', 'bilateral',
        'suspicious', 'concerning', 'malignancy', 'cancer', 'tumor'
    ]
    
    moderate_severity_terms = [
        'moderate', 'mild', 'small', 'minimal', 'slight', 'minor',
        'possible', 'likely', 'probable', 'suggest', 'compatible'
    ]
    
    negative_terms = [
        'no', 'normal', 'clear', 'negative', 'unremarkable', 'stable',
        'resolved', 'improved', 'decreased', 'without'
    ]
    
    text_lower = text.lower()
    
    high_severity_count = sum(1 for term in high_severity_terms if term in text_lower)
    moderate_severity_count = sum(1 for term in moderate_severity_terms if term in text_lower)
    negative_count = sum(1 for term in negative_terms if term in text_lower)
    
    severity_score = (high_severity_count * 0.15) + (moderate_severity_count * 0.08) - (negative_count * 0.05)
    severity_score = max(0, min(severity_score, 1.0))
    
    embedding_complexity = min(analysis_result['embedding_std'] / 0.5, 1.0)
    
    biobert_score = (severity_score * 0.6) + (embedding_complexity * 0.4)
    return max(0.1, min(biobert_score, 1.0))

# ============================================================================
# Clinical Feature Extraction
# ============================================================================

def extract_clinical_features(text, patient_age=50):
    """Extract clinical features from text"""
    text_lower = text.lower()
    
    critical_keywords = {
        'bilateral': ['bilateral', 'both lung', 'both sides'],
        'severe': ['severe', 'extensive', 'massive', 'large'],
        'acute': ['acute', 'emergency', 'urgent', 'immediate', 'critical'],
        'suspicious': ['suspicious', 'concerning', 'malignancy', 'cancer', 'tumor'],
        'fracture': ['fracture', 'broken', 'rib fracture']
    }
    
    critical_counts = {}
    for category, keywords in critical_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        critical_counts[category] = count
    
    pathology_terms = [
        'pneumonia', 'pneumothorax', 'effusion', 'consolidation',
        'atelectasis', 'edema', 'mass', 'lesion', 'nodule',
        'cardiomegaly', 'opacity', 'infiltrate'
    ]
    pathology_count = sum(1 for term in pathology_terms if term in text_lower)
    
    high_severity_indicators = ['severe', 'extensive', 'large', 'massive', 'critical']
    moderate_severity_indicators = ['moderate', 'mild', 'small', 'minimal']
    negative_indicators = ['normal', 'clear', 'no acute', 'unremarkable', 'no evidence']
    
    high_severity_count = sum(1 for ind in high_severity_indicators if ind in text_lower)
    moderate_severity_count = sum(1 for ind in moderate_severity_indicators if ind in text_lower)
    negative_count = sum(1 for ind in negative_indicators if ind in text_lower)
    
    laterality_bilateral = 1 if 'bilateral' in text_lower or 'both' in text_lower else 0
    laterality_unilateral = 1 if ('right' in text_lower or 'left' in text_lower) and not laterality_bilateral else 0
    
    age_risk = min(patient_age / 100.0, 1.0)
    
    features = {
        'critical_bilateral': critical_counts['bilateral'],
        'critical_severe': critical_counts['severe'],
        'critical_acute': critical_counts['acute'],
        'critical_suspicious': critical_counts['suspicious'],
        'critical_fracture': critical_counts['fracture'],
        'pathology_count': pathology_count,
        'high_severity_count': high_severity_count,
        'moderate_severity_count': moderate_severity_count,
        'negative_count': negative_count,
        'laterality_bilateral': laterality_bilateral,
        'laterality_unilateral': laterality_unilateral,
        'age_risk': age_risk,
        'patient_age': patient_age
    }
    
    return features

def calculate_clinical_features_score(features):
    """Calculate clinical features risk score"""
    score = 0.0
    
    score += features['critical_bilateral'] * 0.15
    score += features['critical_severe'] * 0.15
    score += features['critical_acute'] * 0.12
    score += features['critical_suspicious'] * 0.18
    score += features['critical_fracture'] * 0.10
    score += min(features['pathology_count'] * 0.08, 0.4)
    score += features['high_severity_count'] * 0.10
    score += features['moderate_severity_count'] * 0.05
    score -= features['negative_count'] * 0.08
    score += features['laterality_bilateral'] * 0.10
    score += features['age_risk'] * 0.10
    
    return max(0.1, min(score, 1.0))

# ============================================================================
# Feature Engineering
# ============================================================================

def create_unified_feature_vector(chexpert_findings, chexpert_score, 
                                   biobert_result, biobert_score,
                                   clinical_features, clinical_score):
    """
    Create a unified 14-feature vector combining all analysis results.
    This matches the notebook's feature engineering pipeline.
    """
    # 1. CheXpert features (3 features)
    chexpert_positive_count = sum(1 for v in chexpert_findings.values() if v > 0 and v != chexpert_findings.get('No Finding', 0))
    chexpert_normalized_count = chexpert_positive_count / 14.0  # Normalize by max possible
    
    # Check for high-risk findings
    high_risk_findings = ['Pneumonia', 'Pneumothorax', 'Lung Lesion', 'Consolidation']
    chexpert_high_risk = max([chexpert_findings.get(finding, 0) for finding in high_risk_findings])
    
    # 2. BioBERT features (3 features)
    biobert_mean = biobert_result['embedding_mean'] if biobert_result else 0.0
    biobert_std = biobert_result['embedding_std'] if biobert_result else 0.0
    
    # 3. Clinical features (8 features)
    clinical_bilateral = min(clinical_features['critical_bilateral'] / 3.0, 1.0)
    clinical_severe = min(clinical_features['critical_severe'] / 3.0, 1.0)
    clinical_acute = min(clinical_features['critical_acute'] / 3.0, 1.0)
    clinical_pathology = min(clinical_features['pathology_count'] / 10.0, 1.0)
    clinical_high_sev = min(clinical_features['high_severity_count'] / 5.0, 1.0)
    clinical_negative = min(clinical_features['negative_count'] / 5.0, 1.0)
    clinical_age = clinical_features['age_risk']
    
    # Create feature vector (must match FEATURE_COLUMNS order)
    feature_vector = {
        'chexpert_score': chexpert_score,
        'chexpert_positive_findings': chexpert_normalized_count,
        'chexpert_high_risk_present': chexpert_high_risk,
        'biobert_score': biobert_score,
        'biobert_embedding_mean': biobert_mean,
        'biobert_embedding_std': biobert_std,
        'clinical_score': clinical_score,
        'clinical_bilateral': clinical_bilateral,
        'clinical_severe': clinical_severe,
        'clinical_acute': clinical_acute,
        'clinical_pathology_count': clinical_pathology,
        'clinical_high_severity': clinical_high_sev,
        'clinical_negative_indicators': clinical_negative,
        'clinical_age_risk': clinical_age
    }
    
    return feature_vector

def normalize_features(feature_vector, scaler=None):
    """
    Normalize features to 0-1 range using MinMaxScaler.
    If scaler is None, creates and fits a new one (for training).
    """
    # Convert to array in correct order
    feature_array = np.array([feature_vector[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
    
    if scaler is None:
        # Create new scaler (for training data)
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(feature_array)
    else:
        # Use existing scaler (for prediction)
        normalized = scaler.transform(feature_array)
    
    return normalized[0], scaler

def predict_with_xgboost(xgb_model, normalized_features):
    """
    Generate XGBoost predictions and probabilities.
    Returns the HIGH risk probability as the XGBoost score.
    """
    if xgb_model is None:
        # Fallback: use average if model not available
        return None, None
    
    try:
        # Reshape for prediction
        features_reshaped = normalized_features.reshape(1, -1)
        
        # Get predictions
        prediction = xgb_model.predict(features_reshaped)[0]
        probabilities = xgb_model.predict_proba(features_reshaped)[0]
        
        # XGBoost score is the HIGH risk probability (index 2)
        xgb_score = probabilities[2]
        
        return xgb_score, probabilities
    except Exception as e:
        print(f"⚠ XGBoost prediction error: {e}")
        return None, None

# ============================================================================
# Risk Classification
# ============================================================================

def classify_risk(risk_percentage):
    """Classify risk based on percentage"""
    if risk_percentage < 30:
        return 'LOW'
    elif risk_percentage <= 70:
        return 'MEDIUM'
    else:
        return 'HIGH'

def generate_summary(text, findings, risk_category, risk_score):
    """Generate medical condition summary"""
    # Get positive findings
    positive_findings = [k for k, v in findings.items() if v > 0 and k != 'No Finding']
    
    if not positive_findings:
        finding_text = "No significant pathological findings detected."
    elif len(positive_findings) == 1:
        finding_text = f"Finding detected: {positive_findings[0]}."
    else:
        finding_text = f"Findings detected: {', '.join(positive_findings[:3])}."
    
    # Detect severity terms
    severity_terms = []
    text_lower = text.lower()
    if any(term in text_lower for term in ['severe', 'extensive', 'massive']):
        severity_terms.append('severe presentation')
    if 'bilateral' in text_lower:
        severity_terms.append('bilateral involvement')
    if any(term in text_lower for term in ['urgent', 'emergency', 'immediate']):
        severity_terms.append('requires immediate attention')
    
    severity_text = ' - ' + ', '.join(severity_terms) if severity_terms else ''
    
    # Risk description
    risk_descriptions = {
        'HIGH': 'High cancer risk detected. Immediate clinical correlation and follow-up recommended.',
        'MEDIUM': 'Moderate cancer risk detected. Clinical follow-up and monitoring recommended.',
        'LOW': 'Low cancer risk. Routine monitoring suggested.'
    }
    
    summary = f"{finding_text}{severity_text} {risk_descriptions[risk_category]}"
    
    return summary

# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_report(image_path, biobert_model, biobert_tokenizer, xgb_model=None, scaler=None):
    """Process a radiology report image and generate risk assessment"""
    
    print("\n" + "="*80)
    print("CANCER RISK STRATIFICATION ANALYSIS")
    print("="*80)
    
    # Step 1: OCR extraction
    raw_text = extract_text_from_image(image_path)
    
    # Step 2: Text cleaning
    print("Cleaning extracted text...")
    cleaned_text = clean_text(raw_text)
    print(f"✓ Cleaned text: {len(cleaned_text)} characters")
    
    # Step 3: CheXpert labeling
    print("Analyzing with CheXpert labeler...")
    chexpert_findings = chexpert_label_extractor(cleaned_text)
    chexpert_score = calculate_chexpert_score(chexpert_findings)
    print(f"✓ CheXpert score: {chexpert_score*100:.1f}%")
    
    # Step 4: BioBERT analysis
    print("Analyzing with BioBERT...")
    biobert_result = biobert_analyze(cleaned_text, biobert_model, biobert_tokenizer)
    biobert_score = calculate_biobert_score(cleaned_text, biobert_result)
    print(f"✓ BioBERT score: {biobert_score*100:.1f}%")
    
    # Step 5: Clinical feature extraction
    print("Extracting clinical features...")
    clinical_features = extract_clinical_features(cleaned_text)
    clinical_score = calculate_clinical_features_score(clinical_features)
    print(f"✓ Clinical score: {clinical_score*100:.1f}%")
    
    # Step 6: Feature Engineering - Create unified feature vector
    print("Creating unified feature vector...")
    feature_vector = create_unified_feature_vector(
        chexpert_findings, chexpert_score,
        biobert_result, biobert_score,
        clinical_features, clinical_score
    )
    print(f"✓ Feature vector created: {len(feature_vector)} features")
    
    # Step 7: Normalize features
    print("Normalizing features...")
    normalized_features, used_scaler = normalize_features(feature_vector, scaler)
    print(f"✓ Features normalized")
    
    # Step 8: XGBoost prediction
    xgboost_score = None
    xgb_probabilities = None
    
    if xgb_model is not None:
        print("Generating XGBoost predictions...")
        xgboost_score, xgb_probabilities = predict_with_xgboost(xgb_model, normalized_features)
        
        if xgboost_score is not None:
            print(f"✓ XGBoost score: {xgboost_score*100:.1f}%")
            print(f"  Probabilities: LOW={xgb_probabilities[0]:.3f}, MEDIUM={xgb_probabilities[1]:.3f}, HIGH={xgb_probabilities[2]:.3f}")
        else:
            print("⚠ XGBoost prediction failed, using fallback")
    else:
        print("⚠ XGBoost model not available, using fallback averaging")
    
    # Fallback if XGBoost fails or is unavailable
    if xgboost_score is None:
        xgboost_score = (chexpert_score + biobert_score + clinical_score) / 3.0
        print(f"✓ XGBoost score (fallback): {xgboost_score*100:.1f}%")
    
    # Step 9: Calculate ensemble score
    ensemble_score = (
        biobert_score * ENSEMBLE_WEIGHTS['biobert'] +
        chexpert_score * ENSEMBLE_WEIGHTS['chexpert'] +
        xgboost_score * ENSEMBLE_WEIGHTS['xgboost'] +
        clinical_score * ENSEMBLE_WEIGHTS['clinical']
    )
    ensemble_percentage = ensemble_score * 100
    
    # Step 10: Classify risk
    risk_category = classify_risk(ensemble_percentage)
    
    # Step 11: Generate summary
    summary = generate_summary(cleaned_text, chexpert_findings, risk_category, ensemble_percentage)
    
    print("\n" + "-"*80)
    print(f"FINAL RISK SCORE: {ensemble_percentage:.1f}% ({risk_category})")
    print("-"*80)
    
    # Prepare feature importance data
    feature_importance = None
    if xgb_model is not None:
        try:
            importance_values = xgb_model.feature_importances_
            feature_importance = {
                FEATURE_COLUMNS[i]: float(importance_values[i]) 
                for i in range(len(FEATURE_COLUMNS))
            }
        except:
            pass
    
    return {
        'image_path': image_path,
        'risk_score': ensemble_percentage,
        'risk_category': risk_category,
        'chexpert_score': chexpert_score * 100,
        'biobert_score': biobert_score * 100,
        'xgboost_score': xgboost_score * 100,
        'clinical_score': clinical_score * 100,
        'summary': summary,
        'positive_findings': [k for k, v in chexpert_findings.items() if v > 0],
        'xgb_probabilities': xgb_probabilities,
        'feature_vector': feature_vector,
        'normalized_features': normalized_features.tolist(),
        'feature_importance': feature_importance,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def save_results_to_csv(results, output_path='risk_assessment_output.csv'):
    """Save results to CSV file"""
    df = pd.DataFrame([{
        'Timestamp': results['timestamp'],
        'Image_Path': results['image_path'],
        'Risk_Score_%': f"{results['risk_score']:.1f}",
        'Risk_Category': results['risk_category'],
        'CheXpert_Score_%': f"{results['chexpert_score']:.1f}",
        'BioBERT_Score_%': f"{results['biobert_score']:.1f}",
        'XGBoost_Score_%': f"{results['xgboost_score']:.1f}",
        'Clinical_Score_%': f"{results['clinical_score']:.1f}",
        'Positive_Findings': ', '.join(results['positive_findings'][:5]),
        'Medical_Summary': results['summary']
    }])
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    print("\nCSV Contents:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Display feature importance if available
    if results.get('feature_importance'):
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS (XGBoost)")
        print("="*80)
        
        # Sort by importance
        sorted_features = sorted(
            results['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\nTop 10 Most Important Features for Risk Prediction:")
        print("-"*80)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"{i:2d}. {feature:35s} {importance:.4f} {bar}")
        
        print("="*80)
    
    # Display XGBoost probabilities if available
    if results.get('xgb_probabilities') is not None:
        probs = results['xgb_probabilities']
        print("\n" + "="*80)
        print("XGBOOST RISK PROBABILITIES")
        print("="*80)
        print(f"  LOW Risk:    {probs[0]*100:5.1f}%")
        print(f"  MEDIUM Risk: {probs[1]*100:5.1f}%")
        print(f"  HIGH Risk:   {probs[2]*100:5.1f}%")
        print("="*80)

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 risk_model.py <path_to_report_image>")
        print("\nExample:")
        print("  python3 risk_model.py /path/to/xray_report.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validate image path
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("INITIALIZING CANCER RISK STRATIFICATION SYSTEM")
    print("="*80 + "\n")
    
    # Initialize BioBERT
    biobert_model, biobert_tokenizer = initialize_biobert()
    if biobert_model is None or biobert_tokenizer is None:
        print("Error: Failed to initialize BioBERT model")
        sys.exit(1)
    
    # Initialize XGBoost model (may be None if not trained yet)
    xgb_model, scaler = initialize_xgboost_model()
    
    if xgb_model is None:
        print("\n" + "="*80)
        print("⚠ XGBoost Model Not Found")
        print("="*80)
        print("No pre-trained XGBoost model detected.")
        print("The system will use a simplified fallback scoring method.")
        print("\nTo train an XGBoost model:")
        print("  1. Create a training dataset with multiple reports")
        print("  2. Run the training script (see documentation)")
        print("  3. The trained model will be saved as 'xgboost_risk_model.pkl'")
        print("\nContinuing with fallback method...")
        print("="*80 + "\n")
    
    # Process the report
    results = process_report(image_path, biobert_model, biobert_tokenizer, xgb_model, scaler)
    
    # Save results to CSV
    output_filename = f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_to_csv(results, output_filename)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
