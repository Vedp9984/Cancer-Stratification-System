# Quick Start Guide - Cancer Risk Stratification Model

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Step 2: Install Python Dependencies
```bash
cd /home/ved_maurya/sem5/DSI/NNN_for_Cancer/Code/backend
pip install -r requirements.txt
```

### Step 3: Run the Model
```bash
# Create test images
python3 create_test_images.py

# Analyze a report
python3 risk_model.py sample_report_high_risk.png
```

## 📊 Usage Examples

### Example 1: Analyze High-Risk Report
```bash
python3 risk_model.py sample_report_high_risk.png
```
**Expected Output:**
- Risk Score: ~75-85%
- Risk Category: HIGH
- Findings: Pneumonia, Pleural Effusion, Cardiomegaly

### Example 2: Analyze Low-Risk Report
```bash
python3 risk_model.py sample_report_low_risk.png
```
**Expected Output:**
- Risk Score: ~15-25%
- Risk Category: LOW
- Findings: No Finding

### Example 3: Analyze Your Own Report
```bash
python3 risk_model.py /path/to/your/xray_report.jpg
```

## 📄 Output Format

The script generates a CSV file: `risk_assessment_YYYYMMDD_HHMMSS.csv`

**Columns:**
- `Timestamp`: When analysis was performed
- `Risk_Score_%`: Overall risk score (0-100)
- `Risk_Category`: LOW, MEDIUM, or HIGH
- `CheXpert_Score_%`, `BioBERT_Score_%`, `XGBoost_Score_%`, `Clinical_Score_%`: Component scores
- `Positive_Findings`: List of detected conditions
- `Medical_Summary`: Human-readable assessment

## 🔍 Understanding Results

### Risk Categories
| Category | Score Range | Meaning |
|----------|-------------|---------|
| **LOW** | 0-30% | Routine monitoring |
| **MEDIUM** | 30-70% | Follow-up recommended |
| **HIGH** | 70-100% | Immediate attention required |

### Component Scores
- **BioBERT (40%)**: Medical text understanding
- **CheXpert (30%)**: Pathology detection
- **XGBoost (20%)**: ML prediction
- **Clinical (10%)**: Severity assessment

## ⚡ Performance Tips

1. **Use Clear Images**: Better image quality = more accurate OCR
2. **GPU Acceleration**: Install PyTorch with CUDA for faster processing
3. **Batch Processing**: Process multiple reports by creating a wrapper script

## 🐛 Troubleshooting

### "Tesseract not found"
```bash
# Install Tesseract OCR (see Step 1 above)
which tesseract  # Verify installation
```

### "Import Error: No module named 'transformers'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
The model will automatically fall back to CPU if GPU memory is insufficient.

### Low Accuracy on Custom Images
- Ensure image is clear and readable
- Check that text is horizontal (not rotated)
- Verify image contains medical report text

## 📚 Additional Resources

- Full documentation: `README_risk_model.md`
- Original notebook: `mock_model.ipynb`
- Dependencies: `requirements.txt`

## ⚠️ Important Notes

1. **Not for Clinical Use**: This is a research/educational tool
2. **Validate Results**: All assessments should be reviewed by medical professionals
3. **Privacy**: Ensure patient data is handled according to regulations (HIPAA, GDPR, etc.)

## 🎯 Next Steps

1. Test with provided sample images
2. Analyze your own X-ray reports
3. Review the detailed README for advanced usage
4. Check the original notebook for model architecture details

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full README (`README_risk_model.md`)
3. Examine the notebook (`mock_model.ipynb`) for implementation details

---

**Ready to get started?** Run the commands in Step 3! 🚀
