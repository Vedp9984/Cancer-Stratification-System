#!/usr/bin/env python3
"""
Test script for risk_model.py
Creates a sample text image for testing the OCR and risk assessment pipeline
"""

import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_sample_report_image(output_path='sample_report.png'):
    """Create a sample X-ray report image with text"""
    
    # Sample radiology report text
    report_text = """
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
    """
    
    # Create image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a common font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw text
    y_position = 30
    for line in report_text.strip().split('\n'):
        line = line.strip()
        if line:
            # Wrap long lines
            wrapped_lines = textwrap.wrap(line, width=70)
            for wrapped_line in wrapped_lines:
                draw.text((30, y_position), wrapped_line, fill='black', font=font)
                y_position += 22
        else:
            y_position += 10
    
    # Save image
    img.save(output_path)
    print(f"✓ Sample report image created: {output_path}")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Format: PNG")
    return output_path

def create_low_risk_report(output_path='sample_report_low_risk.png'):
    """Create a low-risk sample report"""
    
    report_text = """
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
    """
    
    width, height = 800, 500
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    y_position = 30
    for line in report_text.strip().split('\n'):
        line = line.strip()
        if line:
            wrapped_lines = textwrap.wrap(line, width=70)
            for wrapped_line in wrapped_lines:
                draw.text((30, y_position), wrapped_line, fill='black', font=font)
                y_position += 22
        else:
            y_position += 10
    
    img.save(output_path)
    print(f"✓ Low-risk sample created: {output_path}")
    return output_path

def main():
    """Create sample images for testing"""
    print("Creating sample X-ray report images for testing...\n")
    print("="*60)
    
    # Create high-risk sample
    high_risk_path = create_sample_report_image('sample_report_high_risk.png')
    
    # Create low-risk sample
    low_risk_path = create_low_risk_report('sample_report_low_risk.png')
    
    print("\n" + "="*60)
    print("✓ Sample images created successfully!")
    print("\nTest the risk model with:")
    print(f"  python3 risk_model.py {high_risk_path}")
    print(f"  python3 risk_model.py {low_risk_path}")
    print("\nNote: Make sure to install dependencies first:")
    print("  pip install -r requirements.txt")
    print("  sudo apt-get install tesseract-ocr  # For Ubuntu/Debian")

if __name__ == "__main__":
    main()
