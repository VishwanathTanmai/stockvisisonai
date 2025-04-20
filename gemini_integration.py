import os
import google.generativeai as genai
from PIL import Image
import io
import base64
import streamlit as st
from utils import pil_image_to_byte_array

def initialize_gemini_api():
    """
    Initialize Google Gemini API with key.
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        st.error("Google API key not found. Please add it to secrets as GOOGLE_API_KEY")
        return False
    
    genai.configure(api_key=api_key)
    return True

def encode_image_base64(image):
    """
    Encode a PIL image to base64 for Gemini API.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def get_gemini_analysis(prediction, image, patient_symptoms=None):
    """
    Get detailed analysis of a skin condition using Google Gemini API with enhanced
    multi-modal capabilities that incorporate both image data and patient symptoms.
    
    Args:
        prediction: Predicted skin condition
        image: PIL Image object of the skin condition
        patient_symptoms: Optional string of patient-reported symptoms
        
    Returns:
        str: Detailed analysis text with explainable AI components
    """
    # Try to initialize Gemini API
    api_available = initialize_gemini_api()
    
    if not api_available:
        # Return a fallback analysis if API isn't available
        return get_fallback_analysis(prediction)
    
    try:
        # Prepare the image for Gemini
        image_byte_array = pil_image_to_byte_array(image)
        
        # Create a Gemini model instance with settings for detailed analysis
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more factual responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1500,  # Increased for more comprehensive analysis
        }
        
        # Use Gemini Pro Vision for image analysis
        model = genai.GenerativeModel(
            model_name="gemini-pro-vision",
            generation_config=generation_config
        )
        
        # Create an enhanced prompt that incorporates explainable AI elements
        # and multi-modal integration (if patient symptoms are provided)
        
        symptom_context = ""
        if patient_symptoms:
            symptom_context = f"""
            The patient has also reported the following symptoms:
            "{patient_symptoms}"
            
            Please incorporate this symptom information in your analysis and note if the symptoms
            are consistent with the visual diagnosis or if they suggest alternative considerations.
            """
        
        prompt = f"""
        This is an image of a skin condition that has been identified as {prediction} with high confidence.
        
        {symptom_context}
        
        Please provide a comprehensive analysis in a structured format:
        
        ## Visual Assessment
        - Describe precisely what visual characteristics you can observe in the image
        - Identify the specific features that are most indicative of {prediction}
        - Note any atypical or unusual aspects that might warrant further investigation
        
        ## Diagnostic Considerations
        - Key diagnostic criteria for {prediction}
        - Potential differential diagnoses that should be considered
        - Typical diagnostic tests that might be used to confirm this condition
        
        ## Etiology and Contributing Factors
        - Primary causes and mechanisms of {prediction}
        - Common triggers and exacerbating factors
        - Relevant genetic, environmental, or immunological factors
        
        ## Management Approaches
        - First-line treatments and their mechanisms of action
        - Lifestyle modifications that may improve outcomes
        - Expected timeline for improvement with appropriate treatment
        
        ## Clinical Guidance
        - Specific signs that would warrant immediate medical attention
        - Long-term monitoring considerations
        - Preventive strategies to reduce recurrence
        
        Important: Your analysis should be educational in nature and reflect current medical understanding.
        The analysis should be detailed but accessible to both patients and healthcare providers.
        Include a clear disclaimer about not replacing professional medical diagnosis.
        """
        
        # Get response from Gemini
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_byte_array}])
        
        # Process and return the analysis
        analysis = response.text
        
        # Add enhanced disclaimer
        disclaimer = """
        
        **MEDICAL DISCLAIMER**: 
        This analysis is provided for informational and educational purposes only. It is not intended as a substitute for professional medical advice, diagnosis, or treatment. The image analysis technology used here, while advanced, has limitations and should be considered as a supplementary tool. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of information provided by this system.
        """
        analysis += disclaimer
        
        return analysis
    
    except Exception as e:
        st.error(f"Error getting analysis from Gemini API: {str(e)}")
        st.warning("Using fallback analysis due to API error. For the most accurate results, please try again later or ensure API key is valid.")
        return get_fallback_analysis(prediction)

def get_fallback_analysis(prediction):
    """
    Provide a fallback analysis when the Gemini API is unavailable.
    
    Args:
        prediction: Predicted skin condition
        
    Returns:
        str: Fallback analysis text
    """
    # Fallback information for each condition
    fallback_info = {
        "Acne": """
## Acne Analysis

Acne is a common skin condition characterized by pimples, blackheads, and clogged pores. It typically affects the face, chest, and back.

### Causes and Triggers:
- Excess oil (sebum) production
- Hair follicles clogged by oil and dead skin cells
- Bacteria (Propionibacterium acnes)
- Hormonal changes, especially during puberty
- Diet (some studies suggest high-glycemic foods may worsen acne)
- Stress

### General Treatment Approaches:
- Topical treatments containing benzoyl peroxide, salicylic acid, or retinoids
- Oral antibiotics for moderate to severe acne
- Hormonal treatments for women (such as certain birth control pills)
- Isotretinoin for severe, cystic acne

### When to Consult a Dermatologist:
- If over-the-counter treatments aren't effective after 2-3 months
- If acne is severe, painful, or cystic
- If acne is causing scarring
- If acne is causing psychological distress

**DISCLAIMER**: This analysis is provided for informational purposes only and should not be considered medical advice. Please consult with a qualified healthcare professional for proper diagnosis and treatment.
        """,
        
        "Hyperpigmentation": """
## Hyperpigmentation Analysis

Hyperpigmentation is a condition where patches of skin become darker than the surrounding areas due to excess melanin production.

### Causes and Triggers:
- Sun exposure (sun spots or solar lentigines)
- Hormonal changes (melasma)
- Inflammation (post-inflammatory hyperpigmentation)
- Certain medications
- Genetic factors

### General Treatment Approaches:
- Sun protection (SPF 30+ daily)
- Topical treatments with ingredients like hydroquinone, kojic acid, vitamin C, retinoids
- Chemical peels
- Laser therapy
- Microdermabrasion

### When to Consult a Dermatologist:
- If hyperpigmentation is widespread or severe
- If it appears suddenly or changes rapidly
- If it's accompanied by other symptoms
- If over-the-counter treatments haven't helped after several months

**DISCLAIMER**: This analysis is provided for informational purposes only and should not be considered medical advice. Please consult with a qualified healthcare professional for proper diagnosis and treatment.
        """,
        
        "Nail Psoriasis": """
## Nail Psoriasis Analysis

Nail psoriasis is a manifestation of psoriasis that affects the fingernails and toenails, causing pitting, discoloration, and separation from the nail bed.

### Causes and Triggers:
- Autoimmune reaction where the body attacks its own tissues
- Genetic predisposition
- Stress
- Injury to the nail
- Infections
- Certain medications

### General Treatment Approaches:
- Topical corticosteroids
- Vitamin D analogues
- Retinoids
- Biologics (for severe cases)
- Intralesional steroid injections
- Light therapy

### When to Consult a Dermatologist:
- If nail changes are painful or interfering with daily activities
- If there are signs of infection (swelling, warmth, discharge)
- If nail changes are accompanied by joint pain or skin symptoms
- If there's significant nail deformity

**DISCLAIMER**: This analysis is provided for informational purposes only and should not be considered medical advice. Please consult with a qualified healthcare professional for proper diagnosis and treatment.
        """,
        
        "SJS-TEN": """
## Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS-TEN) Analysis

SJS-TEN is a severe, potentially life-threatening skin reaction usually triggered by medications, causing skin peeling and painful blisters.

### Causes and Triggers:
- Medication reactions (most common with antibiotics, anticonvulsants, NSAIDs)
- Infections (less commonly)
- Genetic factors affecting drug metabolism
- Compromised immune system

### General Treatment Approaches:
- Immediate discontinuation of suspected triggering medication
- Supportive care in a hospital setting, often in burn units
- Fluid and electrolyte management
- Wound care
- Immunoglobulin therapy or cyclosporine (in some cases)
- Pain management

### When to Consult a Doctor IMMEDIATELY:
- This is a medical emergency requiring immediate attention
- Symptoms include widespread rash, blisters, peeling skin
- Mucous membrane involvement (eyes, mouth, genitals)
- Fever, flu-like symptoms preceding rash

**IMPORTANT DISCLAIMER**: SJS-TEN is a medical emergency requiring immediate professional medical attention. This information is provided for educational purposes only. If you suspect SJS-TEN, seek emergency medical care immediately.
        """,
        
        "Vitiligo": """
## Vitiligo Analysis

Vitiligo is a long-term condition where patches of skin lose their color due to the destruction of melanocytes, the cells responsible for skin pigmentation.

### Causes and Triggers:
- Autoimmune disorder (the immune system attacks melanocytes)
- Genetic factors
- Oxidative stress
- Neural factors
- Triggering events like sunburn, emotional stress, or physical injury

### General Treatment Approaches:
- Topical corticosteroids
- Calcineurin inhibitors
- Phototherapy (UVB or PUVA)
- Skin grafting for stable vitiligo
- Depigmentation (for widespread vitiligo)
- Camouflage products and makeup

### When to Consult a Dermatologist:
- When vitiligo first appears
- If areas of depigmentation are spreading rapidly
- If vitiligo is causing emotional or psychological distress
- To discuss treatment options and management strategies

**DISCLAIMER**: This analysis is provided for informational purposes only and should not be considered medical advice. Please consult with a qualified healthcare professional for proper diagnosis and treatment.
        """
    }
    
    return fallback_info.get(prediction, "Detailed analysis unavailable. Please consult with a healthcare professional for information about this condition.")
