import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BartTokenizer

# Set page configuration
st.set_page_config(
    page_title="📊 Conversational Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# Load summarization model
@st.cache_resource
def load_summarization_model():
    try:
        st.write("Debug: Loading summarization model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        raise e

# Load summarization tokenizer
@st.cache_resource
def load_summarization_tokenizer():
    try:
        st.write("Debug: Loading summarization tokenizer...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        return tokenizer
    except Exception as e:
        st.error(f"Error loading summarization tokenizer: {str(e)}")
        raise e

# Load text generation model
@st.cache_resource
def load_llm_model(model_name="gpt2"):
    try:
        st.write(f"Debug: Loading text generation model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generator
    except Exception as e:
        st.error(f"Error loading text generation model: {str(e)}")
        raise e

# Function to query LLM models locally
import traceback

def query_llm(prompt, model_name="gpt2"):
    try:
        # Load the model
        generator = load_llm_model(model_name)
        
        # Debug: Print the prompt
        st.write("Debug: Prompt sent to the model:")
        st.code(prompt)
        
        # Generate response
        response = generator(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        
        # Debug: Print the raw response
        st.write("Debug: Raw response from the model:")
        st.write(response)
        
        return response[0]["generated_text"]
    except Exception as e:
        st.error(f"Error querying the model: {str(e)}")
        st.write("Debug: Full error details:")
        st.write(traceback.format_exc())  # Print the full traceback
        return "Sorry, I couldn't process your request."

# Function to extract and execute Python code from LLM response
def execute_code_from_response(response):
    code_pattern = r"```python(.*?)```"
    code_blocks = re.findall(code_pattern, response, re.DOTALL)
    
    results = []
    
    if code_blocks:
        for code in code_blocks:
            output_buffer = StringIO()
            try:
                local_vars = {
                    'pd': pd,
                    'plt': plt,
                    'sns': sns,
                    'np': np,
                    'df': st.session_state.data
                }
                
                exec(code, local_vars)
                
                if plt.get_fignums():
                    fig = plt.gcf()
                    results.append(("figure", fig))
                    plt.close()
                
                if 'result_df' in local_vars:
                    results.append(("dataframe", local_vars['result_df']))
                
            except Exception as e:
                results.append(("error", str(e)))
    
    return results

# Function to generate data summary using Hugging Face model
def generate_data_summary(df):
    if df.empty:
        return "The dataset is empty. Please upload a valid CSV file."
    
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    describe_str = df.describe().to_string()
    
    columns_str = "Columns: " + ", ".join(df.columns.tolist())
    sample_str = "Sample data:\n" + df.head(3).to_string()
    
    summary_text = f"{columns_str}\n\n{info_str}\n\n{describe_str}\n\n{sample_str}"
    
    # Debug: Print the length of the summary text
    st.write(f"Debug: Length of summary text: {len(summary_text)} characters")
    
    # Tokenize the summary text
    tokenizer = load_summarization_tokenizer()
    tokens = tokenizer.encode(summary_text, truncation=False, return_tensors="pt")
    
    # Truncate the tokens to 1024 tokens (BART's max input length)
    max_tokens = 1024
    if len(tokens[0]) > max_tokens:
        tokens = tokens[:, :max_tokens]
        st.write(f"Debug: Truncated summary text to {max_tokens} tokens")
    
    # Decode the tokens back to text
    truncated_summary_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    # Debug: Print the truncated summary text
    st.write("Debug: Truncated summary text sent to the summarization model:")
    st.code(truncated_summary_text)
    
    try:
        summarizer = load_summarization_model()
        response = summarizer(truncated_summary_text, max_length=150, min_length=30)
        return response[0]["summary_text"]
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.write("Debug: Full error details:")
        st.write(e)
        # Fallback to basic statistics
        return f"Basic Statistics:\n\n{describe_str}\n\nSample Data:\n\n{sample_str}"

# Main layout
st.title("📊 Conversational Data Explorer")
st.markdown("""
Upload your CSV file and chat with your data using natural language!
This app uses Hugging Face models for natural language processing and enhanced analysis.
""")

# Sidebar with file upload and model selection
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    # Model selection
    hf_model = st.selectbox(
        "Select Hugging Face Model",
        [
            "gpt2",  # Smaller model for testing
            "distilgpt2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "google/gemma-7b-it",
            "microsoft/phi-2"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("## How to use")
    st.markdown("""
    1. Upload a CSV file
    2. The app will display a summary of your data
    3. Ask questions about your data in natural language
    4. The AI will respond and generate visualizations
    """)
    
    st.divider()
    st.markdown("## Deployment Info")
    st.info("""
    This app is configured for cloud deployment on:
    - Streamlit Cloud
    - Hugging Face Spaces
    - Other cloud platforms
    
    It uses Hugging Face's models for all AI functionality.
    """)

# Process uploaded file
if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name):
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.session_state.file_name = uploaded_file.name
        
        # Generate data summary
        st.session_state.data_summary = generate_data_summary(data)
        
        # Reset conversation
        st.session_state.conversation = []
        
        # Add system message to conversation
        system_msg = {
            "role": "system", 
            "content": f"Successfully loaded {uploaded_file.name} with {data.shape[0]} rows and {data.shape[1]} columns."
        }
        st.session_state.conversation.append(system_msg)
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Display data and chat interface if data is loaded
if st.session_state.data is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(5), use_container_width=True)
    
    with col2:
        st.subheader("Data Summary")
        if st.session_state.data_summary:
            st.info(st.session_state.data_summary)
        
        st.write("Basic Statistics:")
        st.write(f"Rows: {st.session_state.data.shape[0]}, Columns: {st.session_state.data.shape[1]}")
        
        st.write("Column Types:")
        dtypes = st.session_state.data.dtypes.astype(str).to_dict()
        for col, dtype in dtypes.items():
            st.write(f"- {col}: {dtype}")
    
    st.divider()
    st.subheader("Chat with your Data")
    
    for message in st.session_state.conversation:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"**You:** {content}")
        elif role == "assistant":
            st.markdown(f"**AI:** {content}")
        elif role == "system":
            st.success(content)
    
    chat_input = st.chat_input("Ask about your data (e.g., 'Show me a histogram of age distribution')")
    
    if chat_input:
        user_msg = {"role": "user", "content": chat_input}
        st.session_state.conversation.append(user_msg)
        
        df_info = f"""
        Dataset: {st.session_state.file_name}
        Columns: {', '.join(st.session_state.data.columns.tolist())}
        Shape: {st.session_state.data.shape}
        Sample data: 
        {st.session_state.data.head(3).to_string()}
        """
        
        prompt = f"""
You are a data analysis assistant. Help the user explore their dataset and provide insights.

Here's information about the current dataset:
{df_info}

When appropriate, generate Python code using pandas, matplotlib, or seaborn to visualize data or perform analysis.
Always wrap any Python code in ```python code blocks. Name any resulting dataframe as 'result_df' if you want it to be displayed.

User question: {chat_input}

First, think through what the user is asking for. Then respond with an explanation and relevant Python code if visualization or computation is needed.
"""
        
        response = query_llm(prompt, model_name=hf_model)
        
        code_results = execute_code_from_response(response)
        
        assistant_msg = {"role": "assistant", "content": response}
        st.session_state.conversation.append(assistant_msg)
        
        st.markdown(f"**AI:** {response}")
        
        for result_type, result in code_results:
            if result_type == "figure":
                st.pyplot(result)
            elif result_type == "dataframe":
                st.dataframe(result)
            elif result_type == "error":
                st.error(f"Error executing code: {result}")
        
        st.rerun()
else:
    st.info("👈 Please upload a CSV file to get started.")
    st.markdown("""
    ### Features:
    - Natural language interaction with your data
    - Automated data visualization
    - Insights generation using AI
    - Support for common data analysis tasks
    
    This application uses Hugging Face's models to create an intuitive data exploration experience.
    Deploy it easily on Streamlit Cloud or Hugging Face Spaces!
    """)