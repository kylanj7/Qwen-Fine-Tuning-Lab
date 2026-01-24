import streamlit as st
from llama_cpp import Llama
import time
from typing import Iterator, Dict, Any
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Fine-Tune LLM Test Lab",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with fixed color scheme
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
        background-color: #0e1117;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #4a90e2;
    }
    .assistant-message {
        background-color: #1a1a1a;
        border-left: 4px solid #00ff88;
    }
    .metric-container {
        background-color: #000000;
        color: #00ff88;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #00ff88;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    .stMarkdown {
        color: #ffffff;
    }
    /* Fix input box visibility */
    .stChatInputContainer {
        background-color: #262730;
    }
    /* Improve sidebar visibility */
    .css-1d391kg {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str, **kwargs) -> Llama:
    """Load the GGUF model with caching and GPU support"""
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            # Get GPU layers from kwargs
            n_gpu_layers = kwargs.get('n_gpu_layers', -1)
            
            # Create model with explicit GPU settings
            llm = Llama(
                model_path=model_path,
                n_ctx=kwargs.get('n_ctx', 2048),
                n_threads=kwargs.get('n_threads', 8),
                n_gpu_layers=n_gpu_layers,  # -1 means offload all layers to GPU
                n_batch=512,  # Batch size for prompt processing
                verbose=True,  # Enable verbose to see GPU loading
                use_mlock=True,  # Keep model in RAM
                use_mmap=True,  # Use memory mapping
                # CUDA-specific settings
                tensor_split=None,  # Can specify GPU split if multiple GPUs
                main_gpu=0,  # Use first GPU
            )
        
        # Display GPU info
        st.success(f"✅ Model loaded successfully!")
        if n_gpu_layers != 0:
            st.info(f"🎮 GPU Layers: {n_gpu_layers} (-1 = all layers on GPU)")
        else:
            st.warning("⚠️ Running on CPU only")
        
        return llm
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Make sure you have llama-cpp-python installed with CUDA support:")
        st.code("CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir")
        return None

GGUF_MODELS_DIR = "models/gguf"

def find_gguf_models() -> list:
    """Find all GGUF models in the models/gguf directory"""
    gguf_dir = Path(GGUF_MODELS_DIR)
    if not gguf_dir.exists():
        return []
    gguf_files = list(gguf_dir.glob("*.gguf"))
    return sorted([str(f) for f in gguf_files])

def format_prompt(message: str, chat_history: list, system_prompt: str = "") -> str:
    """Format the prompt for the model"""
    formatted = ""
    
    if system_prompt:
        formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    # Only include recent history to avoid context overflow
    recent_history = chat_history[-6:]  # Last 3 exchanges
    
    for msg in recent_history:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    formatted += f"<|im_start|>user\n{message}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    
    return formatted

def stream_response(
    llm: Llama,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[Dict[str, Any]]:
    """Stream the model response with metrics"""
    
    start_time = time.time()
    tokens_generated = 0
    
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=True,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    for output in stream:
        if 'choices' in output and len(output['choices']) > 0:
            choice = output['choices'][0]
            if 'text' in choice:
                tokens_generated += 1
                elapsed_time = time.time() - start_time
                tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                
                yield {
                    'text': choice['text'],
                    'tokens_generated': tokens_generated,
                    'elapsed_time': elapsed_time,
                    'tokens_per_sec': tokens_per_sec
                }

def main():
    st.title("⚗️ Fine-Tune LLM Test Lab")
    st.markdown("Test and evaluate fine-tuned language models")
    
    # Sidebar - Model Selection and Configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Find available GGUF models
        gguf_models = find_gguf_models()
        
        if not gguf_models:
            st.error(f"No GGUF models found in {GGUF_MODELS_DIR}/")
            st.info("Please run train.py and merge_and_convert_gguff.py to generate a GGUF model.")
            st.stop()
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            gguf_models,
            index=0,
            help="Choose the GGUF model to use for inference"
        )
        
        st.divider()
        
        # System Prompt
        st.subheader("💬 System Prompt")
        system_prompt = st.text_area(
            "System Prompt",
            value=" ",
            height=200,
            help="Set the behavior and context for the model"
        )
        
        st.divider()
        
        # Model Loading Configuration
        st.subheader("🚀 Model Loading")
        
        # GPU Configuration
        st.markdown("**GPU Configuration**")
        use_gpu = st.checkbox("Enable GPU Acceleration", value=True, help="Offload layers to GPU")
        
        if use_gpu:
            n_gpu_layers = -1
            st.success("✅ GPU enabled - all layers offloaded to GPU")
        else:
            n_gpu_layers = 0
            st.warning("⚠️ GPU disabled - model will run on CPU only")
        
        n_ctx = st.number_input(
            "Context Length",
            min_value=512,
            max_value=8192,
            value=2048,
            step=512,
            help="Maximum context window size"
        )
        
        n_threads = st.slider(
            "CPU Threads",
            min_value=1,
            max_value=32,
            value=8,
            help="Number of CPU threads (for CPU layers and prompt processing)"
        )
        
        st.divider()
        
        # Generation Parameters
        st.subheader("📈 Generation Parameters")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=4096,
            max_value=16384,
            value=4096,
            step=1,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.01,
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        
        top_p = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            help="Cumulative probability cutoff for token selection"
        )
        
        top_k = st.slider(
            "Top K",
            min_value=0,
            max_value=100,
            value=40,
            step=1,
            help="Number of top tokens to consider. 0 = disabled"
        )
        
        st.divider()
        
        # System Info
        with st.expander("ℹ️ System Information"):
            st.markdown("**CUDA Status**")
            try:
                import torch
                if torch.cuda.is_available():
                    st.success(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
                    st.info(f"CUDA Version: {torch.version.cuda}")
                else:
                    st.warning("⚠️ CUDA not available")
            except ImportError:
                st.info("PyTorch not installed (not required for llama-cpp)")
            
            st.markdown("**llama-cpp-python Info**")
            st.code(f"Model: {selected_model}\nContext: {n_ctx}\nGPU Layers: {n_gpu_layers}")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Load the model
    llm = load_model(
        selected_model,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers
    )
    
    if llm is None:
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        css_class = "user-message" if role == "user" else "assistant-message"
        with st.container():
            st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
            st.markdown(f"**{role.capitalize()}**")
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Test your new LLM..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.container():
            st.markdown('<div class="chat-message user-message">', unsafe_allow_html=True)
            st.markdown("**User**")
            st.markdown(prompt)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Format the prompt
        formatted_prompt = format_prompt(prompt, st.session_state.messages[:-1], system_prompt)
        
        # Display assistant response with streaming
        with st.container():
            st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
            st.markdown("**Assistant**")
            
            # Create placeholders
            response_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            full_response = ""
            
            # Stream the response
            last_update = time.time()
            for chunk in stream_response(
                llm,
                formatted_prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
            ):
                full_response += chunk['text']
                response_placeholder.markdown(full_response + "▌")
                
                # Update metrics every 0.1 seconds to reduce flickering
                current_time = time.time()
                if current_time - last_update > 0.1:
                    metrics_placeholder.markdown(
                        f'<div class="metric-container">'
                        f'⚡ <strong>{chunk["tokens_per_sec"]:.1f}</strong> tok/s &nbsp;│&nbsp; '
                        f'🔢 <strong>{chunk["tokens_generated"]}</strong> tokens &nbsp;│&nbsp; '
                        f'⏱️ <strong>{chunk["elapsed_time"]:.1f}</strong>s'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    last_update = current_time
            
            # Final display without cursor
            response_placeholder.markdown(full_response)
            
            # Final metrics
            metrics_placeholder.markdown(
                f'<div class="metric-container">'
                f'✅ Complete: <strong>{chunk["tokens_per_sec"]:.1f}</strong> tok/s &nbsp;│&nbsp; '
                f'<strong>{chunk["tokens_generated"]}</strong> tokens &nbsp;│&nbsp; '
                f'<strong>{chunk["elapsed_time"]:.1f}</strong>s total'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
