"""
Chatbot module for the skin disease prediction system.
Uses transformers library for the chatbot functionality.
"""

import os
# Import these only if available
try:
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import streamlit as st

class SkinDiseaseAssistant:
    """
    A chatbot assistant specialized in skin disease information.
    Uses a transformer-based model for generating responses.
    """
    
    def __init__(self):
        """Initialize the chatbot."""
        self.history = []
        
        # Initialize transformer model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            self.using_transformers = True
        except Exception as e:
            st.warning(f"Could not initialize transformer model: {str(e)}")
            self.using_transformers = False
            self.initialize_fallback_responses()
        except Exception as e:
            st.warning(f"Could not initialize transformer model: {str(e)}")
            self.using_transformers = False
            self.initialize_fallback_responses()
    
    def initialize_fallback_responses(self):
        """Initialize fallback responses for when the model is not available."""
        self.skin_condition_info = {
            "acne": {
                "description": "A skin condition that occurs when hair follicles are clogged with oil and dead skin cells.",
                "causes": "Excess oil production, bacteria, inflammation, and clogged pores.",
                "treatments": "Topical medications, oral antibiotics, isotretinoin, and lifestyle adjustments.",
                "prevention": "Regular cleansing, avoiding touching your face, and proper diet."
            },
            "hyperpigmentation": {
                "description": "A condition where patches of skin become darker than the surrounding areas.",
                "causes": "Sun exposure, inflammation, hormonal changes, and certain medications.",
                "treatments": "Topical treatments, chemical peels, laser therapy, and sun protection.",
                "prevention": "Sun protection, avoiding picking at skin, and treating inflammation promptly."
            },
            "nail psoriasis": {
                "description": "A condition that causes nail changes like pitting, discoloration, and separation from the nail bed.",
                "causes": "Autoimmune response that accelerates skin cell growth in the nail area.",
                "treatments": "Topical treatments, steroid injections, systemic medications, and biologics.",
                "prevention": "Keeping nails trimmed, avoiding trauma to nails, and systemic treatments."
            },
            "sjs-ten": {
                "description": "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe skin reactions, typically to medications.",
                "causes": "Medication reactions, particularly to antibiotics, anti-seizure drugs, and some pain relievers.",
                "treatments": "Immediate medication discontinuation, supportive care, wound care, and sometimes immunosuppressive therapy.",
                "prevention": "Avoiding known trigger medications and genetic testing for susceptibility."
            },
            "vitiligo": {
                "description": "A condition that causes loss of skin color in patches due to destruction of pigment-producing cells.",
                "causes": "Autoimmune disorder, genetic factors, triggers like sunburn or emotional stress.",
                "treatments": "Topical corticosteroids, calcineurin inhibitors, phototherapy, and surgical techniques.",
                "prevention": "Sun protection, stress management, and early treatment of developing patches."
            }
        }
        
        self.general_responses = {
            "greeting": ["Hello! I'm your skin health assistant. How can I help you today?", 
                         "Hi there! Do you have questions about skin conditions?",
                         "Welcome! I'm here to help with skin health information."],
            "farewell": ["Take care of your skin! Goodbye!", 
                         "Wishing you healthy skin! Goodbye!",
                         "Feel free to return if you have more questions. Goodbye!"],
            "unknown": ["I'm not sure about that. Could you ask about specific skin conditions like acne, hyperpigmentation, vitiligo, nail psoriasis, or SJS-TEN?",
                        "I don't have information on that topic. I can help with common skin conditions though.",
                        "I'm specialized in skin conditions. Could you rephrase your question about skin health?"]
        }
    
    def generate_response(self, user_input):
        """
        Generate a response based on the user input.
        
        Args:
            user_input: String containing the user's message
            
        Returns:
            String with the bot's response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_input})
        
        # Generate response using transformer model if available
        if self.using_transformers:
            try:
                # This would use the transformer model if it was properly initialized
                response = "I would use the transformer model here if it was working properly."
            except Exception as e:
                # Fallback to rule-based responses
                response = self.get_fallback_response(user_input)
        else:
            # Use rule-based responses
            response = self.get_fallback_response(user_input)
        
        # Add bot response to history
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def get_fallback_response(self, user_input):
        """Generate a rule-based response."""
        user_input = user_input.lower()
        
        # Check for greetings
        if any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
            import random
            return random.choice(self.general_responses["greeting"])
        
        # Check for farewells
        if any(word in user_input for word in ["bye", "goodbye", "see you", "farewell"]):
            import random
            return random.choice(self.general_responses["farewell"])
        
        # Check for specific skin conditions
        for condition, info in self.skin_condition_info.items():
            if condition in user_input:
                if "what" in user_input and any(word in user_input for word in ["is", "are"]):
                    return info["description"]
                elif "cause" in user_input:
                    return f"Causes of {condition}: {info['causes']}"
                elif any(word in user_input for word in ["treat", "cure", "help"]):
                    return f"Treatments for {condition}: {info['treatments']}"
                elif "prevent" in user_input:
                    return f"Prevention for {condition}: {info['prevention']}"
                else:
                    return f"{info['description']} It is caused by {info['causes']} Treatments include {info['treatments']}"
        
        # General questions about skin
        if "skin" in user_input:
            if "care" in user_input or "routine" in user_input:
                return "A good skin care routine includes cleansing, moisturizing, and sun protection. For specific skin conditions, additional targeted treatments may be necessary."
            elif "protect" in user_input or "sun" in user_input:
                return "Sun protection is crucial for skin health. Use broad-spectrum sunscreen with SPF 30 or higher, wear protective clothing, and seek shade during peak sun hours."
            elif "diet" in user_input or "food" in user_input or "eat" in user_input:
                return "A balanced diet rich in fruits, vegetables, lean proteins, and omega-3 fatty acids can promote skin health. Staying hydrated is also important."
        
        # If nothing specific was matched
        import random
        return random.choice(self.general_responses["unknown"])
    
    def reset_chat(self):
        """Reset the chat history."""
        self.history = []
    
    def get_history(self):
        """Return the chat history."""
        return self.history


def create_chatbot_ui():
    """Create the chatbot UI components for Streamlit."""
    # Initialize chatbot in session state if it doesn't exist
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = SkinDiseaseAssistant()
    
    # Header
    st.markdown("## Skin Health Assistant")
    st.markdown("Ask questions about skin conditions, treatments, and prevention.")
    
    # Initialize messages in session state if they don't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Skin Health Assistant. How can I help you today?"}]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about skin conditions..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.write(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Reset button
    if st.button("Reset Chat"):
        st.session_state.chatbot.reset_chat()
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Skin Health Assistant. How can I help you today?"}]
        st.rerun()