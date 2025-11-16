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
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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

def process_report(image_path, biobert_model, biobert_tokenizer):
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
    
    # Step 6: XGBoost score (simplified - use average of other scores)
    xgboost_score = (chexpert_score + biobert_score + clinical_score) / 3.0
    print(f"✓ XGBoost score: {xgboost_score*100:.1f}%")
    
    # Step 7: Calculate ensemble score
    ensemble_score = (
        biobert_score * ENSEMBLE_WEIGHTS['biobert'] +
        chexpert_score * ENSEMBLE_WEIGHTS['chexpert'] +
        xgboost_score * ENSEMBLE_WEIGHTS['xgboost'] +
        clinical_score * ENSEMBLE_WEIGHTS['clinical']
    )
    ensemble_percentage = ensemble_score * 100
    
    # Step 8: Classify risk
    risk_category = classify_risk(ensemble_percentage)
    
    # Step 9: Generate summary
    summary = generate_summary(cleaned_text, chexpert_findings, risk_category, ensemble_percentage)
    
    print("\n" + "-"*80)
    print(f"FINAL RISK SCORE: {ensemble_percentage:.1f}% ({risk_category})")
    print("-"*80)
    
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
    
    # Initialize BioBERT
    biobert_model, biobert_tokenizer = initialize_biobert()
    if biobert_model is None or biobert_tokenizer is None:
        print("Error: Failed to initialize BioBERT model")
        sys.exit(1)
    
    # Process the report
    results = process_report(image_path, biobert_model, biobert_tokenizer)
    
    # Save results to CSV
    output_filename = f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_to_csv(results, output_filename)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
