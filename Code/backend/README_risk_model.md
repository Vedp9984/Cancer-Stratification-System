# Cancer Risk Stratification Model

## Overview
This Python script analyzes radiology report images using OCR and NLP techniques to generate cancer risk scores (0-100%) with risk categorization (Low/Medium/High).

## Features
- **OCR Text Extraction**: Extracts text from X-ray report images using Tesseract
- **Multi-Model Analysis**:
  - **BioBERT** (40%): Medical text contextual understanding
  - **CheXpert** (30%): Structured pathology finding extraction
  - **XGBoost** (20%): Machine learning ensemble prediction
  - **Clinical Features** (10%): Rule-based severity assessment
- **Risk Classification**: Categorizes risk as LOW (<30%), MEDIUM (30-70%), or HIGH (>70%)
- **CSV Output**: Generates comprehensive risk assessment report

## Installation

### Prerequisites
1. **Tesseract OCR Engine** (required for OCR functionality):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Python 3.8+**

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note**: First run will download BioBERT model (~420MB) automatically.

## Usage

### Basic Usage
```bash
python3 risk_model.py <path_to_report_image>
```

### Example
```bash
python3 risk_model.py /path/to/chest_xray_report.jpg
```

### Supported Image Formats
- JPG/JPEG
- PNG
- TIFF
- BMP
- PDF (single page)

## Output

### CSV File
The script generates a CSV file with the following structure:

| Column | Description |
|--------|-------------|
| Timestamp | Date and time of analysis |
| Image_Path | Path to input image |
| Risk_Score_% | Overall risk score (0-100%) |
| Risk_Category | LOW, MEDIUM, or HIGH |
| CheXpert_Score_% | CheXpert labeler score |
| BioBERT_Score_% | BioBERT analysis score |
| XGBoost_Score_% | XGBoost model score |
| Clinical_Score_% | Clinical features score |
| Positive_Findings | Detected pathological findings |
| Medical_Summary | Human-readable risk summary |

### Example Output
```
Timestamp: 2025-11-16 14:30:45
Image_Path: chest_report.jpg
Risk_Score_%: 78.5
Risk_Category: HIGH
CheXpert_Score_%: 82.1
BioBERT_Score_%: 76.3
XGBoost_Score_%: 80.2
Clinical_Score_%: 75.8
Positive_Findings: Pneumonia, Pleural Effusion, Consolidation
Medical_Summary: Findings detected: Pneumonia, Pleural Effusion, Consolidation - severe presentation, bilateral involvement. High cancer risk detected. Immediate clinical correlation and follow-up recommended.
```

## Risk Classification Thresholds

- **LOW Risk** (0-30%): Routine monitoring suggested
- **MEDIUM Risk** (30-70%): Clinical follow-up and monitoring recommended
- **HIGH Risk** (70-100%): Immediate clinical correlation required

## Model Architecture

### Ensemble Weights
- **BioBERT**: 40% - Contextual understanding of medical terminology
- **CheXpert**: 30% - Structured extraction of 14 pathology categories
- **XGBoost**: 20% - Machine learning prediction based on combined features
- **Clinical Features**: 10% - Rule-based severity indicators

### CheXpert Pathology Categories
1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

## Technical Details

### Processing Pipeline
1. **OCR Extraction**: Extract text from image using Tesseract
2. **Text Cleaning**: Normalize and clean extracted text
3. **CheXpert Analysis**: Keyword-based pathology detection
4. **BioBERT Analysis**: Generate medical text embeddings
5. **Clinical Feature Extraction**: Extract severity indicators
6. **Ensemble Scoring**: Weighted combination of all methods
7. **Risk Classification**: Categorize based on thresholds
8. **Summary Generation**: Create human-readable report

### Performance Characteristics
- **Processing Time**: 5-15 seconds per report (depends on image size and hardware)
- **GPU Support**: Automatically uses CUDA if available for BioBERT
- **Memory Usage**: ~1-2GB RAM (including BioBERT model)

## Limitations

1. **OCR Quality**: Accuracy depends on image quality and text clarity
2. **Training Data**: Model based on limited sample reports (proof of concept)
3. **Medical Validation**: Results should be validated by medical professionals
4. **Not for Diagnosis**: This is a risk stratification tool, not a diagnostic tool

## Troubleshooting

### "Tesseract not found" Error
Install Tesseract OCR engine (see Prerequisites above)

### CUDA/GPU Issues
Script automatically falls back to CPU if CUDA is unavailable

### Memory Errors
Reduce image size or ensure sufficient RAM (minimum 2GB recommended)

### Import Errors
Run: `pip install -r requirements.txt`

## Development

### Convert Jupyter Notebook to Script
The original notebook can be found at `mock_model.ipynb`

### Extending the Model
- Add more pathology categories in `CHEXPERT_KEYWORDS`
- Adjust ensemble weights in `ENSEMBLE_WEIGHTS`
- Modify risk thresholds in `RISK_THRESHOLDS`
- Train XGBoost model on larger dataset for better accuracy

## References

- **BioBERT**: Biomedical Language Representation Model (dmis-lab/biobert-v1.1)
- **CheXpert**: Stanford CheXpert Dataset and Labeler
- **XGBoost**: Extreme Gradient Boosting Library

## License

This is a proof-of-concept implementation for educational and research purposes.

## Disclaimer

**This tool is for research and educational purposes only. It is not intended for clinical use or medical diagnosis. All results should be reviewed and validated by qualified medical professionals.**
