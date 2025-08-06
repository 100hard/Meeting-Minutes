import os
import json
import requests
from datetime import datetime
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gradio as gr
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_MODEL = "whisper-1"
GPT_OSS = "openai/gpt-oss-20b"

class MeetingMinutesGenerator:
    def __init__(self):
        self.openai_client = None
        self.tokenizer = None
        self.model = None
        self.mode = "lite"  # Default to lite mode
        
    def setup_openai(self, api_key):
        """Setup OpenAI client"""
        try:
            self.openai_client = OpenAI(api_key=api_key)
            # Test the connection
            self.openai_client.models.list()
            logger.info("✅ OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ OpenAI setup failed: {e}")
            return False
    
    def setup_huggingface(self, hf_token):
        """Setup HuggingFace models for Pro mode"""
        try:
            from huggingface_hub import login
            
            login(hf_token, add_to_git_credential=True)
            logger.info("✅ HuggingFace login successful")
            
            # Load tokenizer
            logger.info(f"Loading tokenizer: {GPT_OSS}")
            self.tokenizer = AutoTokenizer.from_pretrained(GPT_OSS, token=hf_token)
            # GPT-OSS uses different special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✅ Tokenizer loaded")
            
            # Check if CUDA is available for quantization
            if torch.cuda.is_available():
                logger.info("🚀 CUDA available - Loading GPT-OSS with quantization")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    GPT_OSS,
                    device_map="auto",
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                    trust_remote_code=True  # GPT-OSS may need this
                )
                logger.info("✅ GPT-OSS model loaded with quantization")
            else:
                logger.info("⚠️ No CUDA - Loading GPT-OSS on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    GPT_OSS,
                    torch_dtype=torch.float32,
                    token=hf_token,
                    trust_remote_code=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ HuggingFace setup failed: {e}")
            return False

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file using OpenAI Whisper"""
        if not self.openai_client:
            return "❌ OpenAI client not initialized. Please check your API key."

        try:
            logger.info(f"Transcribing audio: {audio_file_path}")
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=AUDIO_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            logger.info("✅ Transcription complete")
            return transcription

        except Exception as e:
            error_msg = f"❌ Transcription error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def extract_structured_data(self, text):
        """Extract structured information from transcription"""
        if not text or isinstance(text, str) and text.startswith("❌"):
            return {}

        patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'times': r'\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b',
            'action_words': r'\b(?:will|should|must|need to|action|todo|follow up|assigned|responsible)\b[^.]*',
        }

        extracted = {}
        for key, pattern in patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted[key] = list(set(matches))[:5]  # Limit to 5 matches
            except:
                extracted[key] = []

        return extracted

    def generate_minutes_lite(self, transcription, meeting_type="general"):
        """Generate minutes using OpenAI GPT (Lite Mode)"""
        if not self.openai_client:
            return "❌ OpenAI client not initialized", {}
            
        try:
            logger.info("🚀 Generating minutes with OpenAI GPT (Lite Mode)")
            
            structured_data = self.extract_structured_data(transcription)
            
            system_message = """You are an expert meeting secretary. Create comprehensive meeting minutes that are:
            - Professionally formatted in markdown
            - Structured with clear sections
            - Action-oriented with specific assignments
            - Include metrics where mentioned (dates, numbers, percentages)
            
            Always follow the exact structure requested."""
            
            user_prompt = f"""
            Create detailed meeting minutes following this EXACT structure:

            # Meeting Minutes - {datetime.now().strftime("%B %d, %Y")}

            ## Meeting Overview
            - **Date**: {datetime.now().strftime("%B %d, %Y")}
            - **Type**: {meeting_type.title()} Meeting
            - **Duration**: [Estimate from transcript]

            ## 📋 Key Decisions Made
            [List 3-5 major decisions from the discussion]

            ## 💬 Discussion Highlights
            [Summarize main discussion points]

            ## ✅ Action Items
            | Task | Owner | Deadline | Priority |
            |------|-------|----------|----------|
            [Extract specific tasks with owners]

            ## 📊 Key Metrics & Data Points
            [Any numbers, percentages, or measurements mentioned]

            ## 🔄 Follow-up Required
            [Items needing additional discussion]

            **TRANSCRIPT TO ANALYZE:**
            {transcription[:4000]}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            minutes = response.choices[0].message.content
            logger.info("✅ Minutes generated successfully (Lite Mode)")
            return minutes, structured_data
            
        except Exception as e:
            error_msg = f"❌ Generation error (Lite Mode): {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

    def generate_minutes_pro(self, transcription, meeting_type="general"):
        """Generate minutes using GPT-OSS-20B with quantization (Pro Mode)"""
        if not self.tokenizer or not self.model:
            return "❌ Pro Mode models not initialized. Please check your HuggingFace token.", {}

        try:
            logger.info("🚀 Generating minutes with GPT-OSS-20B (Pro Mode)")
            
            structured_data = self.extract_structured_data(transcription)
            
            # GPT-OSS uses OpenAI-style system/user format
            system_message = """You are an expert meeting secretary. Create comprehensive meeting minutes that are:
            - Professionally formatted in markdown
            - Structured with clear sections
            - Action-oriented with specific assignments
            - Include metrics where mentioned (dates, numbers, percentages)

            Always follow the exact structure requested."""

            user_prompt = f"""
            Create detailed meeting minutes following this EXACT structure:

            # Meeting Minutes - {datetime.now().strftime("%B %d, %Y")}

            ## Meeting Overview
            - **Date**: {datetime.now().strftime("%B %d, %Y")}
            - **Type**: {meeting_type.title()} Meeting
            - **Duration**: [Estimate from transcript]

            ## 📋 Key Decisions Made
            [List 3-5 major decisions from the discussion]

            ## 💬 Discussion Highlights
            [Summarize main discussion points]

            ## ✅ Action Items
            | Task | Owner | Deadline | Priority |
            |------|-------|----------|----------|
            [Extract specific tasks with owners]

            ## 📊 Key Metrics & Data Points
            [Any numbers, percentages, or measurements mentioned]

            ## 🔄 Follow-up Required
            [Items needing additional discussion]

            **TRANSCRIPT TO ANALYZE:**
            {transcription[:3000]}...
            """

            #prompt for GPT-OSS
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]

            # Apply chat template for GPT-OSS
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                combined_prompt = f"System: {system_message}\n\nUser: {user_prompt}\n\nAssistant:"
                inputs = self.tokenizer(combined_prompt, return_tensors="pt")["input_ids"]

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=1500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated minutes - handle GPT-OSS format
            if "# Meeting Minutes" in response:
                minutes_start = response.find("# Meeting Minutes")
                minutes = response[minutes_start:]
            elif "Assistant:" in response:
                minutes = response.split("Assistant:")[-1].strip()
            else:
                # Fallback - take everything after the input
                input_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                minutes = response.replace(input_text, "").strip()

            logger.info("✅ Minutes generated successfully with GPT-OSS-20B")
            return minutes, structured_data

        except Exception as e:
            error_msg = f"❌ Generation error (Pro Mode): {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

def create_gradio_interface():
    """Create a professional, dark-themed Gradio interface"""
    
    generator = MeetingMinutesGenerator()
    
    def process_meeting(audio_file, meeting_type, mode, openai_key, hf_token, progress=gr.Progress()):
        """Process meeting with user-provided keys"""
        
        if audio_file is None:
            return "❌ Please upload an audio file", "{}"
        
        if not openai_key or not openai_key.strip():
            return "❌ Please provide your OpenAI API key", "{}"
        
        try:
            progress(0.1, desc="Setting up...")
            
            # Setup OpenAI (required for both modes)
            if not generator.setup_openai(openai_key.strip()):
                return "❌ Invalid OpenAI API key. Please check and try again.", "{}"
            
            # Setup HuggingFace if Pro mode
            if mode == "pro":
                if not hf_token or not hf_token.strip():
                    return "❌ Pro Mode requires HuggingFace token", "{}"
                
                if not generator.setup_huggingface(hf_token.strip()):
                    return "❌ Invalid HuggingFace token or model loading failed", "{}"
            
            # Transcribe
            progress(0.3, desc="Transcribing audio...")
            transcription = generator.transcribe_audio(audio_file)
            
            if transcription.startswith("❌" or "❌"):
                return transcription, "{}"
            
            # Generate minutes based on mode
            progress(0.7, desc=f"Generating minutes ({mode.title()} Mode)...")
            
            if mode == "lite":
                minutes, structured_data = generator.generate_minutes_lite(transcription, meeting_type)
            else:  # pro mode
                minutes, structured_data = generator.generate_minutes_pro(transcription, meeting_type)
            
            progress(1.0, desc="Complete!")
            return minutes, json.dumps(structured_data, indent=2)
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            logger.error(error_msg)
            return error_msg, "{}"
    
    
    custom_css = """
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global variables */
    :root {
        --primary-bg: #0f1419;
        --secondary-bg: #1a1f2e;
        --tertiary-bg: #242938;
        --accent-bg: #2d3748;
        --primary-text: #e2e8f0;
        --secondary-text: #a0aec0;
        --accent-color: #4299e1;
        --success-color: #48bb78;
        --error-color: #f56565;
        --border-color: #2d3748;
        --hover-border: #4299e1;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .gradio-container {
        background: var(--primary-bg) !important;
        color: var(--primary-text) !important;
        min-height: 100vh !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* Main wrapper */
    .main-wrapper {
        background: var(--secondary-bg) !important;
        margin: 20px !important;
        border-radius: 12px !important;
        padding: 40px !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Header */
    .header {
        text-align: center !important;
        margin-bottom: 40px !important;
        padding: 30px 0 !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    .header h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-text) !important;
        margin: 0 0 12px 0 !important;
        letter-spacing: -0.025em !important;
    }
    
    .header p {
        font-size: 1.1rem !important;
        color: var(--secondary-text) !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }
    
    /* Cards */
    .card {
        background: var(--tertiary-bg) !important;
        border-radius: 8px !important;
        padding: 24px !important;
        margin-bottom: 16px !important;
        border: 1px solid var(--border-color) !important;
        transition: border-color 0.2s ease !important;
    }
    
    .card:hover {
        border-color: var(--hover-border) !important;
    }
    
    .card h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--primary-text) !important;
        margin: 0 0 16px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.875rem !important;
    }
    
    /* Input styling */
    .gradio-textbox input,
    .gradio-dropdown select {
        background: var(--accent-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        color: var(--primary-text) !important;
        transition: all 0.2s ease !important;
    }
    
    .gradio-textbox input::placeholder {
        color: var(--secondary-text) !important;
    }
    
    .gradio-textbox input:focus,
    .gradio-dropdown select:focus {
        border-color: var(--accent-color) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1) !important;
    }
    
    /* Labels */
    .gradio-label label {
        color: var(--primary-text) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 8px !important;
        display: block !important;
    }
    
    /* Audio upload */
    .gradio-audio {
        background: var(--accent-bg) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
        padding: 32px !important;
        transition: all 0.2s ease !important;
        text-align: center !important;
    }
    
    .gradio-audio:hover {
        border-color: var(--accent-color) !important;
        background: rgba(66, 153, 225, 0.05) !important;
    }
    
    /* Radio buttons */
    .gradio-radio {
        background: var(--accent-bg) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .gradio-radio label {
        color: var(--primary-text) !important;
    }
    
    /* Primary button */
    .primary-button {
        background: var(--accent-color) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 16px 24px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin: 24px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
    }
    
    .primary-button:hover {
        background: #3182ce !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(66, 153, 225, 0.25) !important;
    }
    
    /* Output area */
    .output-container {
        background: var(--tertiary-bg) !important;
        border-radius: 8px !important;
        padding: 24px !important;
        margin-top: 24px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Status indicators */
    .status-ready {
        color: var(--success-color) !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 16px !important;
        font-size: 0.9rem !important;
    }
    
    /* Grid layout */
    .input-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 24px !important;
        margin-bottom: 24px !important;
    }
    
    /* Markdown output */
    .gradio-markdown {
        background: var(--accent-bg) !important;
        border-radius: 8px !important;
        padding: 20px !important;
        border: 1px solid var(--border-color) !important;
        color: var(--primary-text) !important;
    }
    
    .gradio-markdown h1,
    .gradio-markdown h2,
    .gradio-markdown h3 {
        color: var(--primary-text) !important;
    }
    
    .gradio-markdown table {
        border-color: var(--border-color) !important;
    }
    
    .gradio-markdown th,
    .gradio-markdown td {
        border-color: var(--border-color) !important;
        color: var(--primary-text) !important;
    }
    
    /* Code blocks */
    .gradio-code {
        background: var(--primary-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Progress bar */
    .gradio-progress {
        background: var(--tertiary-bg) !important;
    }
    
    .gradio-progress .progress-bar {
        background: var(--accent-color) !important;
    }
    
    /* Accordions */
    .gradio-accordion {
        background: var(--tertiary-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
    }
    
    .gradio-accordion summary {
        color: var(--primary-text) !important;
        font-weight: 500 !important;
        padding: 16px !important;
    }
    
    .gradio-accordion[open] summary {
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    .gradio-accordion div {
        padding: 16px !important;
        color: var(--secondary-text) !important;
    }
    
    /* Info text */
    .gr-info {
        color: var(--secondary-text) !important;
        font-size: 0.825rem !important;
    }
    
    @media (max-width: 768px) {
        .input-grid {
            grid-template-columns: 1fr !important;
        }
        
        .header h1 {
            font-size: 2rem !important;
        }
        
        .main-wrapper {
            margin: 10px !important;
            padding: 20px !important;
        }
        
        .card {
            padding: 16px !important;
        }
    }
    
    /* Hide Gradio footer */
    .gradio-container .footer {
        display: none !important;
    }
    
    /* Clean up spacing */
    .gradio-container > div {
        gap: 0 !important;
    }
    
    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-bg) !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color) !important;
    }
    """
    
    # Create the interface
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate
        ),
        title="Meeting Minutes Generator",
        css=custom_css
    ) as interface:
        
        with gr.Column(elem_classes="main-wrapper"):
            # Header
            with gr.Row(elem_classes="header"):
                gr.HTML("""
                <div class="header">
                    <h1>Meeting Minutes Generator</h1>
                    <p>Professional AI-powered transcription and documentation</p>
                </div>
                """)
            
            # Main input section
            with gr.Row(elem_classes="input-grid"):
                # Left column - Setup
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<h3>API Configuration</h3>')
                        
                        openai_key = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-proj-...",
                            type="password",
                            info="Required for transcription and processing"
                        )
                        
                        hf_token = gr.Textbox(
                            label="HuggingFace Token (Optional)",
                            placeholder="hf_...",
                            type="password",
                            info="Only needed for Pro mode"
                        )
                    
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<h3>Processing Mode</h3>')
                        
                        mode = gr.Radio(
                            choices=[
                                ("Lite Mode (Fast)", "lite"),
                                ("Pro Mode (Advanced)", "pro")
                            ],
                            value="lite",
                            label="Select Mode",
                            info="Lite: GPT-3.5 | Pro: GPT-OSS-20B"
                        )
                
                # Right column - Input
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<h3>Audio Upload</h3>')
                        
                        audio_input = gr.Audio(
                            label="Meeting Recording",
                            type="filepath",
                            sources=["upload"],
                            show_label=False
                        )
                        
                        meeting_type = gr.Dropdown(
                            choices=[
                                ("General Meeting", "general"),
                                ("Project Review", "project"),
                                ("Executive Session", "executive")
                            ],
                            value="general",
                            label="Meeting Type",
                            info="Select the type of meeting"
                        )
            
            # Generate button
            process_btn = gr.Button(
                "Generate Meeting Minutes",
                elem_classes="primary-button",
                variant="primary",
                size="lg"
            )
            
            # Output section
            with gr.Group(elem_classes="output-container"):
                gr.HTML("""
                <div class="status-ready">
                    <span>●</span> Ready to process your meeting recording
                </div>
                """)
                
                minutes_output = gr.Markdown(
                    value="**Upload your audio file and click generate to create professional meeting minutes.**",
                    show_label=False
                )
                
                with gr.Accordion("Extracted Data", open=False):
                    structured_output = gr.Code(
                        label="Structured Data (JSON)",
                        language="json",
                        value="{}",
                        lines=6
                    )
            
            # Info section
            with gr.Row():
                with gr.Accordion("API Key Setup", open=False):
                    gr.Markdown("""
                    **OpenAI API Key:**
                    1. Visit [platform.openai.com](https://platform.openai.com)
                    2. Sign up or login to your account
                    3. Navigate to API Keys and create new key
                    4. Copy the key (starts with `sk-`)
                    
                    **HuggingFace Token (Pro Mode only):**
                    1. Visit [huggingface.co](https://huggingface.co)
                    2. Sign up or login to your account
                    3. Go to Settings → Access Tokens
                    4. Create new token with Read permissions
                    5. Copy token (starts with `hf_`)
                    """)
                
                with gr.Accordion("Mode Comparison", open=False):
                    gr.Markdown("""
                    | Feature | Lite Mode | Pro Mode |
                    |---------|-----------|----------|
                    | **Speed** | Fast (1-2 min) | Slower (3-5 min) |
                    | **Quality** | Excellent | Superior |
                    | **Requirements** | OpenAI key only | OpenAI + HF token |
                    | **AI Model** | GPT-3.5 Turbo | GPT-OSS-20B |
                    | **Cost** | Low | Medium |
                    | **Railway Support** | ✅ Recommended | ⚠️ Limited |
                    """)
                
                with gr.Accordion("Railway Deployment Info", open=False):
                    gr.Markdown("""
                    **For Railway Deployment:**
                    - ✅ Lite Mode is **recommended** for Railway
                    - ⚠️ Pro Mode may face memory/timeout limitations on Railway
                    - Railway free tier: 512MB RAM, Pro tier: up to 8GB RAM
                    - For Pro Mode, consider upgrading to Railway Pro plan
                    """)
        
        # Connect the processing function
        process_btn.click(
            fn=process_meeting,
            inputs=[audio_input, meeting_type, mode, openai_key, hf_token],
            outputs=[minutes_output, structured_output],
            show_progress=True
        )
    
    return interface

if __name__ == "__main__":
    logger.info("Starting Meeting Minutes Generator...")
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    
    interface = create_gradio_interface()
    interface.launch(
        server_name=host,
        server_port=port,
        share=False,
        debug=False,
        show_error=True
    )