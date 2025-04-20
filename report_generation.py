from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image as ReportImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image

def generate_pdf_report(patient_name, image, prediction, confidence, analysis_text, 
                        texture_features=None, color_info=None, comparison_image=None,
                        patient_symptoms=None):
    """
    Generate a PDF report for the skin disease prediction.
    
    Args:
        patient_name: Patient name
        image: Original PIL image
        prediction: Predicted skin condition
        confidence: Model confidence score
        analysis_text: Detailed analysis text
        texture_features: Optional texture analysis results
        color_info: Optional color analysis results
        comparison_image: Optional comparison image (before/after)
        patient_symptoms: Optional string of patient-reported symptoms
        
    Returns:
        bytes: PDF report as bytes
    """
    buffer = io.BytesIO()
    
    # Create a PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.navy,
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("Skin Disease Analysis Report", title_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Patient info and date
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"<b>Patient:</b> {patient_name}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {now}", normal_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Prediction result
    story.append(Paragraph("Diagnosis", section_style))
    data = [
        ["Predicted Condition", prediction],
        ["Confidence", f"{confidence:.1f}%"]
    ]
    table = Table(data, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25*inch))
    
    # Image
    story.append(Paragraph("Patient Image", section_style))
    # Save image to bytes buffer
    img_buffer = io.BytesIO()
    # Resize image to fit report page
    report_width = 5*inch
    img_width, img_height = image.size
    img_height = int(report_width * img_height / img_width)
    resized_img = image.resize((int(report_width), img_height))
    resized_img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    # Add image to report
    img = ReportImage(img_buffer, width=report_width, height=img_height)
    story.append(img)
    story.append(Spacer(1, 0.25*inch))
    
    # Patient Symptoms (if provided)
    if patient_symptoms:
        story.append(Paragraph("Patient-Reported Symptoms", section_style))
        symptom_text = f'<i>"{patient_symptoms}"</i>'
        story.append(Paragraph(symptom_text, normal_style))
        story.append(Spacer(1, 0.25*inch))
    
    # Analysis
    story.append(Paragraph("Detailed Analysis", section_style))
    story.append(Paragraph(analysis_text, normal_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Texture analysis if available
    if texture_features:
        story.append(Paragraph("Texture Analysis", section_style))
        
        # Create a figure for texture properties
        fig, ax = plt.subplots(figsize=(6, 3))
        metrics = [
            texture_features["contrast"], 
            texture_features["dissimilarity"],
            texture_features["homogeneity"], 
            texture_features["energy"],
            texture_features["correlation"]
        ]
        metric_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
        y_pos = np.arange(len(metric_names))
        
        ax.barh(y_pos, metrics, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_title('Texture Properties')
        
        # Save figure to a buffer
        texture_buffer = io.BytesIO()
        fig.savefig(texture_buffer, format='png', bbox_inches='tight')
        texture_buffer.seek(0)
        plt.close(fig)
        
        # Add figure to the report
        texture_img = ReportImage(texture_buffer, width=5*inch, height=2.5*inch)
        story.append(texture_img)
        
        # Add texture description
        texture_description = get_texture_report(texture_features)
        story.append(Paragraph(texture_description, normal_style))
        story.append(Spacer(1, 0.25*inch))
    
    # Color analysis if available
    if color_info:
        story.append(Paragraph("Color Analysis", section_style))
        
        # Create a figure for color distribution
        fig, ax = plt.subplots(figsize=(6, 3))
        hex_colors = color_info["hex_colors"]
        color_percentages = color_info["color_percentages"]
        
        # Create pie chart of dominant colors
        ax.pie(color_percentages, colors=hex_colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Color Distribution')
        
        # Save figure to a buffer
        color_buffer = io.BytesIO()
        fig.savefig(color_buffer, format='png', bbox_inches='tight')
        color_buffer.seek(0)
        plt.close(fig)
        
        # Add figure to the report
        color_img = ReportImage(color_buffer, width=4*inch, height=3*inch)
        story.append(color_img)
        
        # Add color description
        uniformity = color_info["uniformity_score"]
        story.append(Paragraph(f"<b>Color Uniformity Score:</b> {uniformity:.1f}%", normal_style))
        
        # Add color description if available
        if "description" in color_info:
            story.append(Paragraph(color_info["description"], normal_style))
        
        story.append(Spacer(1, 0.25*inch))
    
    # Comparison image if available
    if comparison_image:
        story.append(Paragraph("Image Comparison", section_style))
        
        # Save comparison image to buffer
        comp_buffer = io.BytesIO()
        comparison_image.save(comp_buffer, format='JPEG')
        comp_buffer.seek(0)
        
        # Add comparison image to report
        comp_width = 6*inch
        comp_height = int(comp_width * comparison_image.height / comparison_image.width)
        comp_img = ReportImage(comp_buffer, width=comp_width, height=comp_height)
        story.append(comp_img)
        story.append(Spacer(1, 0.25*inch))
    
    # Footer with disclaimer
    disclaimer = (
        "DISCLAIMER: This report is generated for informational purposes only and does not "
        "constitute medical advice. The predictions and analyses are based on computer algorithms "
        "and should not replace professional medical diagnosis. Please consult with a qualified "
        "healthcare professional for proper diagnosis and treatment of any medical condition."
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )
    story.append(Paragraph(disclaimer, disclaimer_style))
    
    # Build the PDF
    doc.build(story)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def get_texture_report(texture_features):
    """Generate a descriptive text about texture features for the report"""
    contrast = texture_features["contrast"]
    homogeneity = texture_features["homogeneity"]
    energy = texture_features["energy"]
    
    report = ["<b>Texture Analysis Summary:</b>"]
    
    # Contrast interpretation
    if contrast > 10:
        report.append("- High contrast indicates significant texture variations, which may be associated with irregular skin surfaces.")
    elif contrast > 5:
        report.append("- Moderate contrast in the texture, showing some variations in the skin surface.")
    else:
        report.append("- Low contrast suggesting a relatively smooth and uniform skin surface.")
    
    # Homogeneity interpretation
    if homogeneity > 0.9:
        report.append("- The texture shows high homogeneity, indicating consistent patterns across the skin area.")
    elif homogeneity > 0.7:
        report.append("- Moderate homogeneity with some variations in texture patterns.")
    else:
        report.append("- Low homogeneity suggesting significant irregularities or different texture regions.")
    
    # Energy interpretation
    if energy > 0.5:
        report.append("- High energy value indicates orderly and regular texture patterns.")
    elif energy > 0.2:
        report.append("- Moderate energy suggesting a mix of ordered and random texture elements.")
    else:
        report.append("- Low energy indicating more random or disordered texture patterns.")
    
    return "<br/>".join(report)

def encode_pdf_to_base64(pdf_data):
    """
    Encode PDF data to base64 for displaying in Streamlit.
    
    Args:
        pdf_data: PDF in bytes
        
    Returns:
        str: Base64 encoded string
    """
    return base64.b64encode(pdf_data).decode('utf-8')

def get_pdf_download_link(pdf_data, filename="skin_analysis_report.pdf"):
    """
    Create an HTML download link for the PDF.
    
    Args:
        pdf_data: PDF in bytes
        filename: Name for the downloaded file
        
    Returns:
        str: HTML code for the download link
    """
    b64 = encode_pdf_to_base64(pdf_data)
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href