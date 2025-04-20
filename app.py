import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import io
import os
import pandas as pd
import time
import base64

from model import load_vit_model, predict_disease, get_class_labels
from utils import preprocess_image, display_predictions, create_heatmap
from gemini_integration import get_gemini_analysis
from visualization import plot_accuracy_curve, plot_loss_curve, plot_feature_extraction, overlay_heatmap
from image_comparison import compare_images, create_comparison_image, highlight_roi
from texture_analysis import analyze_texture, get_texture_description, segment_texture_regions
from color_analysis import analyze_colors, get_skin_tone_description, create_color_map
from report_generation import generate_pdf_report, get_pdf_download_link
from chatbot import create_chatbot_ui

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Prediction App",
    page_icon="🔬",
    layout="wide"
)

# App title and description
st.title("Skin Disease Prediction System")
st.markdown("""
This application uses advanced AI models including Vision Transformers and BERT to analyze and predict 
skin conditions from uploaded images. It can identify five common skin conditions:
- Acne
- Hyperpigmentation
- Nail Psoriasis
- SJS-TEN (Stevens-Johnson Syndrome - Toxic Epidermal Necrolysis)
- Vitiligo
""")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Home", "Predict", "Advanced Analysis", "Image Comparison", "Chat Assistant", "Model Performance", "About"]
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    with st.spinner("Loading model... This might take a moment."):
        st.session_state.model = load_vit_model()
        st.session_state.class_labels = get_class_labels()

if 'history' not in st.session_state:
    st.session_state.history = []

# Home page
if app_mode == "Home":
    st.markdown("## Multi-Modal Skin Disease Prediction System")
    st.markdown("""
    Welcome to our advanced multi-modal skin disease prediction system. This application integrates image analysis with 
    patient-reported symptoms to provide more accurate and comprehensive skin condition predictions.
    
    ### How it works:
    1. Upload an image of the affected skin area
    2. Enter your symptoms for more accurate multi-modal analysis
    3. Our AI model analyses both the image and text data
    4. Google Gemini API provides detailed real-time analysis
    5. View regions of interest with explainable AI visualization
    
    ### Key Features:
    - **Multi-Modal Analysis**: Combines image data with patient-reported symptoms
    - **Real-time Gemini AI Analysis**: Powered by Google's advanced API
    - **Explainable AI Visualization**: See which regions influence the diagnosis
    - **Advanced Image Analysis**: Texture patterns, color distributions, and ROI detection
    - **Comprehensive Reports**: PDF reports with both image and symptom assessment
    
    ### Supported Conditions:
    - **Acne**: Inflammatory skin condition with pimples and clogged pores
    - **Hyperpigmentation**: Darkening of patches of skin
    - **Nail Psoriasis**: Affects fingernails and toenails with pitting and discoloration
    - **SJS-TEN**: Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis - severe skin reaction
    - **Vitiligo**: Loss of skin color in patches
    """)
    
    # Display performance metrics on the home page based on real-time data
    st.markdown("### Real-time Model Performance")
    
    if 'model' in st.session_state and hasattr(st.session_state.model, 'accuracy_history') and len(st.session_state.model.accuracy_history) > 0:
        # Real metrics if we have data
        current_accuracy = st.session_state.model.accuracy_history[-1] * 100
        current_loss = st.session_state.model.loss_history[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Current Accuracy", value=f"{current_accuracy:.1f}%")
        with col2:
            st.metric(label="Current Loss", value=f"{current_loss:.3f}")
        with col3:
            # Count total predictions
            predictions_count = len(st.session_state.model.accuracy_history)
            st.metric(label="Total Predictions", value=str(predictions_count))
            
        st.info("These metrics reflect the current performance of the model based on real-time predictions. Make more predictions to refine these metrics.")
    else:
        # Initial state without predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Model Status", value="Ready")
        with col2:
            st.metric(label="Predictions", value="0")
        
        st.info("No predictions have been made yet. Upload an image on the Predict page to start using the system and see real-time performance metrics.")

# Prediction page
elif app_mode == "Predict":
    st.markdown("## Upload an image for skin disease prediction")
    
    # Create two columns for a more organized layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # More robust file uploader with try-except block
        try:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="predict_page_uploader")
        except Exception as e:
            st.error(f"Error in file uploader: {str(e)}")
            uploaded_file = None
    
    with col2:
        # Explain the multi-modal approach
        st.info("This system uses a multi-modal approach combining image analysis with optional symptom description for more accurate results.")
    
    # Option to include textual symptom description for multi-modal analysis
    st.markdown("### Patient Symptoms (Optional)")
    st.markdown("Describe any symptoms you're experiencing to enhance the analysis:")
    patient_symptoms = st.text_area("Symptoms", placeholder="E.g., itching, burning sensation, pain, duration of symptoms, any triggers you've noticed...", height=100)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add explainability options
        st.markdown("### Analysis Options")
        explainable_ai = st.checkbox("Enable Explainable AI visualization", value=True, 
                                   help="Shows which regions of the image influenced the prediction")
        
        # Add a button to trigger prediction
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Preprocess the image and get predictions using multi-modal approach
                processed_image = preprocess_image(image)
                # Pass patient symptoms for multi-modal analysis
                prediction, confidence, features = predict_disease(
                    st.session_state.model, 
                    processed_image,
                    text_input=patient_symptoms if patient_symptoms else None
                )
                
                # Store prediction in history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                prediction_record = {
                    "timestamp": timestamp,
                    "prediction": prediction,
                    "confidence": confidence,
                    "image": image,
                    "symptoms": patient_symptoms if patient_symptoms else "No symptoms reported"
                }
                st.session_state.history.append(prediction_record)
                
                # Display prediction results
                st.success(f"Analysis complete!")
                
                # Create main columns for results
                main_col1, main_col2 = st.columns([3, 2])
                
                with main_col1:
                    st.markdown("### Prediction Results")
                    display_predictions(prediction, confidence)
                    
                    # Fetch detailed analysis from Gemini API
                    with st.spinner("Generating comprehensive analysis with Gemini AI..."):
                        # Pass patient symptoms to get more personalized analysis
                        gemini_analysis = get_gemini_analysis(
                            prediction, 
                            image, 
                            patient_symptoms=patient_symptoms if patient_symptoms else None
                        )
                        st.markdown("### Detailed Analysis")
                        st.markdown(gemini_analysis)
                
                with main_col2:
                    # Show feature visualization with explainability
                    st.markdown("### Feature Visualization")
                    fig = plot_feature_extraction(features, prediction)
                    st.pyplot(fig)
                    
                    # Add explainable AI visualization if enabled
                    if explainable_ai:
                        st.markdown("### Region of Interest")
                        st.markdown("The highlighted areas show regions that influenced the prediction:")
                        
                        # Generate heatmap for explainability
                        heatmap = create_heatmap(features)
                        heatmap_overlay = overlay_heatmap(image, heatmap, alpha=0.6)
                        st.image(heatmap_overlay, caption="Regions of Interest Highlighting", use_column_width=True)
                        
                        # Add ROI detection with auto-detection
                        roi_image = highlight_roi(image, bbox=None, auto_detect=True)
                        st.image(roi_image, caption="Automatic Region Detection", use_column_width=True)
    
    # Display prediction history
    if st.session_state.history:
        st.markdown("## Prediction History")
        for i, record in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Prediction {i+1} - {record['timestamp']} - {record['prediction']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(record['image'], width=150)
                with col2:
                    st.markdown(f"**Prediction:** {record['prediction']}")
                    st.markdown(f"**Confidence:** {record['confidence']:.2f}%")

# Advanced Analysis page
elif app_mode == "Advanced Analysis":
    st.markdown("## Advanced Skin Analysis Tools")
    
    # More robust file uploader with try-except block
    try:
        uploaded_file = st.file_uploader("Choose an image for advanced analysis...", type=["jpg", "jpeg", "png"], key="advanced_analysis_uploader")
    except Exception as e:
        st.error(f"Error in file uploader: {str(e)}")
        uploaded_file = None
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Create tabs for different analysis types
        analysis_tabs = st.tabs(["Texture Analysis", "Color Profiling", "ROI Detection", "Report Generation"])
        
        with analysis_tabs[0]:  # Texture Analysis
            st.markdown("### Texture Analysis")
            st.markdown("Analyze texture patterns in the skin image to identify irregularities and characteristics.")
            
            if st.button("Analyze Texture"):
                with st.spinner("Analyzing texture features..."):
                    # Perform texture analysis
                    texture_features, texture_fig = analyze_texture(image)
                    
                    # Display texture visualization
                    st.pyplot(texture_fig)
                    
                    # Display texture description
                    st.markdown("### Texture Description")
                    st.markdown(get_texture_description(texture_features))
                    
                    # Display segmented texture regions
                    st.markdown("### Texture Segmentation")
                    segmented_image = segment_texture_regions(image)
                    st.image(segmented_image, caption="Segmented Texture Regions", use_column_width=True)
        
        with analysis_tabs[1]:  # Color Profiling
            st.markdown("### Color Profiling")
            st.markdown("Analyze the color distribution and characteristics of the skin image.")
            
            if st.button("Analyze Colors"):
                with st.spinner("Analyzing color profile..."):
                    # Perform color analysis
                    color_info, color_fig = analyze_colors(image)
                    
                    # Display color visualization
                    st.pyplot(color_fig)
                    
                    # Display color description
                    st.markdown("### Skin Tone Analysis")
                    color_description = get_skin_tone_description(color_info)
                    st.markdown(color_description)
                    
                    # Display color map
                    st.markdown("### Color Map")
                    color_map = create_color_map(image)
                    st.image(color_map, caption="Color-coded Regions", use_column_width=True)
        
        with analysis_tabs[2]:  # ROI Detection
            st.markdown("### Region of Interest (ROI) Detection")
            st.markdown("Automatically detect and highlight regions of interest in the skin image.")
            
            col1, col2 = st.columns(2)
            with col1:
                # Option for auto-detection
                auto_detect = st.checkbox("Auto-detect regions of interest", value=True)
            
            with col2:
                # Manual ROI selection
                if not auto_detect:
                    st.markdown("Manual ROI coordinates:")
                    x1 = st.slider("X1", 0, image.width, int(image.width * 0.25))
                    y1 = st.slider("Y1", 0, image.height, int(image.height * 0.25))
                    x2 = st.slider("X2", 0, image.width, int(image.width * 0.75))
                    y2 = st.slider("Y2", 0, image.height, int(image.height * 0.75))
                    bbox = (x1, y1, x2, y2)
                else:
                    bbox = None
            
            if st.button("Detect ROI"):
                with st.spinner("Detecting regions of interest..."):
                    # Highlight ROI
                    roi_image = highlight_roi(image, bbox, auto_detect)
                    st.image(roi_image, caption="Image with ROI Highlighted", use_column_width=True)
        
        with analysis_tabs[3]:  # Report Generation
            st.markdown("### Report Generation")
            st.markdown("Generate a comprehensive PDF report with analysis results.")
            
            # Patient information form
            st.markdown("#### Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Patient Name", "Anonymous Patient")
            
            # Add patient symptoms input for multi-modal analysis
            st.markdown("#### Patient Symptoms (Optional)")
            patient_symptoms = st.text_area(
                "Describe any symptoms or relevant information",
                placeholder="Enter symptoms, duration, triggers, or other relevant information that might help with the diagnosis...",
                height=100
            )
            
            # Process the image first to get prediction
            if st.button("Generate Report"):
                with st.spinner("Analyzing image and generating report..."):
                    # Preprocess and predict with multi-modal approach
                    processed_image = preprocess_image(image)
                    prediction, confidence, features = predict_disease(
                        st.session_state.model, 
                        processed_image,
                        text_input=patient_symptoms if patient_symptoms else None
                    )
                    
                    # Get detailed analysis with patient symptoms for multi-modal analysis
                    analysis_text = get_gemini_analysis(
                        prediction, 
                        image, 
                        patient_symptoms=patient_symptoms if patient_symptoms else None
                    )
                    
                    # Perform texture and color analysis for the report
                    texture_features, _ = analyze_texture(image)
                    color_info, _ = analyze_colors(image)
                    color_info["description"] = get_skin_tone_description(color_info)
                    
                    # Generate PDF report
                    pdf_data = generate_pdf_report(
                        patient_name=patient_name,
                        image=image,
                        prediction=prediction,
                        confidence=confidence,
                        analysis_text=analysis_text,
                        texture_features=texture_features,
                        color_info=color_info,
                        patient_symptoms=patient_symptoms if patient_symptoms else None
                    )
                    
                    # Create download link
                    st.markdown("### Download Report")
                    st.markdown(get_pdf_download_link(pdf_data), unsafe_allow_html=True)
                    
                    # Display prediction summary
                    st.success(f"Report generated for diagnosis: {prediction} (Confidence: {confidence:.1f}%)")
                    st.info("The report includes a comprehensive multi-modal analysis based on both image data and patient-reported symptoms.")

# Image Comparison page
elif app_mode == "Image Comparison":
    st.markdown("## Image Comparison Tool")
    st.markdown("Compare two skin images to track changes over time or analyze differences between conditions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### First Image (Before)")
        # More robust file uploader with try-except block
        try:
            image1_file = st.file_uploader("Choose first image...", type=["jpg", "jpeg", "png"], key="comparison_image1_uploader")
            if image1_file:
                image1 = Image.open(image1_file).convert('RGB')
                st.image(image1, caption="First Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error in first image uploader: {str(e)}")
            image1_file = None
    
    with col2:
        st.markdown("### Second Image (After)")
        # More robust file uploader with try-except block
        try:
            image2_file = st.file_uploader("Choose second image...", type=["jpg", "jpeg", "png"], key="comparison_image2_uploader")
            if image2_file:
                image2 = Image.open(image2_file).convert('RGB')
                st.image(image2, caption="Second Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error in second image uploader: {str(e)}")
            image2_file = None
    
    # Check if both image variables are defined and not None
    if image1_file and image2_file and 'image1' in locals() and 'image2' in locals():
        st.markdown("### Comparison Options")
        comparison_type = st.radio(
            "Select comparison type:",
            ["Side-by-side with differences", "Overlay with heatmap", "Combined report"]
        )
        
        if st.button("Generate Comparison"):
            with st.spinner("Generating comparison..."):
                try:
                    if comparison_type == "Side-by-side with differences":
                        # Generate side-by-side comparison with differences highlighted
                        fig = compare_images(image1, image2)
                        st.pyplot(fig)
                    
                    elif comparison_type == "Overlay with heatmap":
                        # Create a heatmap from image1
                        processed_image1 = preprocess_image(image1)
                        _, _, features1 = predict_disease(st.session_state.model, processed_image1)
                        
                        # Generate heatmap
                        heatmap = create_heatmap(features1)
                        
                        # Overlay heatmap on image2
                        overlay = overlay_heatmap(image2, heatmap)
                        st.image(overlay, caption="Overlay with Heatmap", use_column_width=True)
                    
                    else:  # Combined report
                        # Generate combined image
                        combined = create_comparison_image(image1, image2, "Before", "After")
                        st.image(combined, caption="Before and After Comparison", use_column_width=True)
                except Exception as e:
                    st.error(f"Error during image comparison: {str(e)}")
                    st.warning("Please ensure both images were uploaded properly before comparison.")
                
                # Only proceed with the rest if no exception was raised
                if comparison_type == "Combined report" and not st.session_state.get('error_occurred', False):
                    try:
                        # Process both images with multi-modal approach (no text input for comparison here)
                        processed_image1 = preprocess_image(image1)
                        processed_image2 = preprocess_image(image2)
                        
                        # Use our vision transformer for predictions
                        prediction1, confidence1, _ = predict_disease(st.session_state.model, processed_image1)
                        prediction2, confidence2, _ = predict_disease(st.session_state.model, processed_image2)
                        
                        # Create comparison table
                        comparison_data = {
                            "Image": ["Before", "After"],
                            "Prediction": [prediction1, prediction2],
                            "Confidence": [f"{confidence1:.1f}%", f"{confidence2:.1f}%"]
                        }
                        df = pd.DataFrame(comparison_data)
                        st.table(df)
                        
                        # Add download option for combined image
                        buf = io.BytesIO()
                        combined.save(buf, format="JPEG")
                        buf.seek(0)
                        
                        st.download_button(
                            label="Download Combined Image",
                            data=buf,
                            file_name="skin_comparison.jpg",
                            mime="image/jpeg"
                        )
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.session_state['error_occurred'] = True

# Chat Assistant page
elif app_mode == "Chat Assistant":
    # Create the chat interface
    create_chatbot_ui()

# Model Performance page
elif app_mode == "Model Performance":
    st.markdown("## Real-time Model Performance Metrics")
    st.markdown("These metrics are updated in real-time as you make predictions with the model.")
    
    tab1, tab2, tab3 = st.tabs(["Accuracy", "Loss", "Confusion Matrix"])
    
    with tab1:
        st.markdown("### Real-time Accuracy Curve")
        acc_fig = plot_accuracy_curve()
        st.pyplot(acc_fig)
        
        # Real-time accuracy metric
        if 'model' in st.session_state and hasattr(st.session_state.model, 'accuracy_history') and len(st.session_state.model.accuracy_history) > 0:
            current_accuracy = st.session_state.model.accuracy_history[-1] * 100
            st.metric(label="Current Model Accuracy", value=f"{current_accuracy:.1f}%")
            st.info("Make more predictions to refine accuracy metrics.")
        
    with tab2:
        st.markdown("### Real-time Loss Curve")
        loss_fig = plot_loss_curve()
        st.pyplot(loss_fig)
        
        # Real-time loss metric
        if 'model' in st.session_state and hasattr(st.session_state.model, 'loss_history') and len(st.session_state.model.loss_history) > 0:
            current_loss = st.session_state.model.loss_history[-1]
            st.metric(label="Current Model Loss", value=f"{current_loss:.3f}")
            st.info("Make more predictions to refine loss metrics.")
        
    with tab3:
        st.markdown("### Real-time Confusion Matrix")
        
        if 'model' not in st.session_state or not hasattr(st.session_state.model, 'confusion_matrix') or st.session_state.model.confusion_matrix.sum() == 0:
            st.warning("No predictions have been made yet. Make some predictions to generate the confusion matrix.")
        else:
            # Get the real confusion matrix from the model
            classes = get_class_labels()
            cm = st.session_state.model.confusion_matrix
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize matrix for percentages (optional)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot the confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title('Real-time Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            
            # Calculate real-time precision, recall, and F1 score
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            class_count = 0
            
            for i in range(len(classes)):
                # Skip classes with no predictions to avoid division by zero
                if cm[:, i].sum() > 0 and cm[i, :].sum() > 0:
                    # Precision: TP / (TP + FP)
                    precision = cm[i, i] / cm[:, i].sum()
                    # Recall: TP / (TP + FN)
                    recall = cm[i, i] / cm[i, :].sum()
                    # F1: 2 * (precision * recall) / (precision + recall)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_sum += precision
                    recall_sum += recall
                    f1_sum += f1
                    class_count += 1
            
            # Calculate averages
            avg_precision = precision_sum / class_count if class_count > 0 else 0
            avg_recall = recall_sum / class_count if class_count > 0 else 0
            avg_f1 = f1_sum / class_count if class_count > 0 else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Precision", value=f"{avg_precision:.1%}")
            with col2:
                st.metric(label="Recall", value=f"{avg_recall:.1%}")
            with col3:
                st.metric(label="F1 Score", value=f"{avg_f1:.1%}")
                
            st.info("These metrics are calculated based on real-time prediction data. Continue making predictions to improve their accuracy.")

# About page
else:
    st.markdown("""
    ## About This Application
    
    This skin disease prediction system was developed using state-of-the-art machine learning technologies:
    
    ### Technologies Used:
    - **Computer Vision**: For advanced image analysis and feature extraction
    - **Machine Learning**: For classification of skin conditions and pattern recognition
    - **Google Gemini API**: For comprehensive analysis and detailed reports
    - **OpenCV & scikit-image**: For texture and color analysis
    - **Streamlit**: For the interactive web interface
    - **ReportLab**: For PDF report generation
    
    ### Core Features:
    
    #### Disease Prediction
    - Image analysis for accurate skin condition classification
    - Confidence scores for predictive reliability
    - Detailed analysis of each condition with causes and treatments
    
    #### Chat Assistant
    - Real-time interactive skin health advisor
    - Answers questions about skin conditions, treatments, and prevention
    - Provides personalized skincare advice
    
    #### Advanced Analysis Tools
    - **Texture Analysis**: Identifies patterns, irregularities, and characteristics in skin texture
    - **Color Profiling**: Analyzes color distribution to identify abnormalities
    - **ROI Detection**: Automatically highlights regions of interest in skin images
    - **Report Generation**: Creates comprehensive PDF reports with all analysis results
    
    #### Image Comparison
    - Side-by-side comparison with difference highlighting
    - Overlay visualization with heatmaps
    - Tracking of condition changes over time
    
    ### How It Works:
    1. **Image Upload**: User uploads a skin condition image
    2. **Feature Extraction**: Key visual features are extracted for analysis
    3. **Multi-level Analysis**: The system performs prediction, texture analysis, color profiling, and ROI detection
    4. **Result Generation**: Results are displayed visually with detailed descriptions
    5. **Report Creation**: Comprehensive reports can be generated and downloaded
    
    ### Supported Conditions:
    - Acne
    - Hyperpigmentation
    - Nail Psoriasis
    - SJS-TEN (Stevens-Johnson Syndrome)
    - Vitiligo
    
    > Note: This application is for informational purposes only and does not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
    """)

# Footer
st.markdown("---")
st.markdown("© 2023 Skin Disease Prediction System | Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice.")
