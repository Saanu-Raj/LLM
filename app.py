import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# Page config
st.set_page_config(
    page_title="TinyLlama Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #4F46E5;
        color: white;
    }
    .chat-message.assistant {
        background-color: white;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# User credentials
USERS = {
    "admin": "admin123",
    "doctor": "doc123",
    "student": "student123"
}

MEDICAL_DISCLAIMER = """
⚠️ **Medical Disclaimer:** This response is for educational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider.
"""

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Login page
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Medical Assistant Login")
        st.markdown("---")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Login", use_container_width=True):
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_b:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **Demo Credentials:**
        - admin / admin123
        - doctor / doc123  
        - student / student123
        """)
    
    st.stop()

# Load model (cached)
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned TinyLlama model with LoRA adapters"""
    try:
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        lora_path = "./tinyllama-medical-lora"
        
        # Check if LoRA weights exist
        if not os.path.exists(lora_path):
            st.error(f"❌ Model not found at {lora_path}")
            st.info("Using base model without fine-tuning...")
            lora_path = None
        
        # Quantization config for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters if available
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            st.success("✅ Fine-tuned model loaded successfully!")
        else:
            st.warning("⚠️ Using base model (not fine-tuned)")
        
        model.eval()
        
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Main app
st.title("🩺 TinyLlama Medical Assistant")
st.caption(f"Logged in as: **{st.session_state.username}**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading status
    if not st.session_state.model_loaded:
        with st.spinner("Loading fine-tuned model..."):
            tokenizer, model = load_model()
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_loaded = True
    
    st.markdown("---")
    
    # Generation parameters
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max New Tokens", 32, 256, 100, 8)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    
    st.markdown("---")
    
    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "What is Paracetamol used for?",
        "Tell me about Ibuprofen",
        "What is Metformin?",
        "Uses of Amoxicillin",
        "What is Atorvastatin for?"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{query}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Logout
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.model_loaded:
                try:
                    # Format prompt
                    formatted_prompt = f"""### Instruction:
{prompt}

### Response:
"""
                    
                    # Tokenize
                    inputs = st.session_state.tokenizer(
                        formatted_prompt,
                        return_tensors="pt"
                    ).to(st.session_state.model.device)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=st.session_state.tokenizer.eos_token_id
                        )
                    
                    # Decode
                    response = st.session_state.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    
                    # Extract only the response part
                    if "### Response:" in response:
                        response = response.split("### Response:")[-1].strip()
                    
                    # Add disclaimer
                    full_response = f"{response}\n\n{MEDICAL_DISCLAIMER}"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            else:
                error_msg = "Model not loaded. Please refresh the page."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.caption("Fine-tuned TinyLlama 1.1B with LoRA on Allopathic Medicine Dataset")