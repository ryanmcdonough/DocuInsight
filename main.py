import streamlit as st
import os
from markitdown import MarkItDown
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import requests
from types import SimpleNamespace
import pdfplumber
from PIL import Image
import base64
import io
import zipfile

# Attempt to import OpenAI; if not available, set OpenAI = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables
load_dotenv()

debug_logs = []
def log_debug(message: str):
    debug_logs.append(message)

class LocalLLMClient:
    def __init__(self, base_url="http://127.0.0.1:1234/v1/chat/completions"):
        self.base_url = base_url
        self.chat = self.Chat(self.base_url)

    class Chat:
        def __init__(self, base_url):
            self.base_url = base_url
            self.completions = LocalLLMClient.Chat.Completions(self.base_url)

        class Completions:
            def __init__(self, base_url):
                self.base_url = base_url

            def create(self, model, messages, temperature=0.7, max_tokens=-1, stream=False):
                log_debug("Preparing to send request to local LLM...")
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }
                log_debug(f"Request Payload: {payload}")
                try:
                    response = requests.post(self.base_url, json=payload, timeout=60)
                    log_debug(f"Request sent to {self.base_url}. HTTP status: {response.status_code}")
                    if response.status_code != 200:
                        log_debug(f"Non-200 status code: {response.status_code}")
                    data = response.json()
                    log_debug(f"Response JSON from LLM: {data}")
                    return self._convert_to_objects(data)
                except requests.Timeout:
                    log_debug("LLM request timed out.")
                    return self._convert_to_objects({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "The LLM request timed out."
                            }
                        }]
                    })
                except Exception as e:
                    log_debug(f"Error making LLM request: {str(e)}")
                    raise e

            def _convert_to_objects(self, data):
                def dict_to_namespace(d):
                    if isinstance(d, dict):
                        for k, v in d.items():
                            d[k] = dict_to_namespace(v)
                        return SimpleNamespace(**d)
                    elif isinstance(d, list):
                        return [dict_to_namespace(x) for x in d]
                    else:
                        return d
                return dict_to_namespace(data)

@st.cache_data
def process_document_cached(file_data, file_name, use_llm, llm_provider, custom_api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name

    try:
        if use_llm:
            if llm_provider == "OpenAI":
                if OpenAI is None:
                    raise RuntimeError("OpenAI library not installed.")
                api_key = custom_api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key is required.")
                client = OpenAI(api_key=api_key)
                llm_client = client
                llm_model = "gpt-4o"
                md = MarkItDown(llm_client=client, llm_model="gpt-4o")
            else:
                llm_client = LocalLLMClient()
                llm_model = "llama-3.1-unhinged-vision-8b"
                md = MarkItDown(llm_client=llm_client, llm_model=llm_model)
        else:
            md = MarkItDown()
            llm_client = None
            llm_model = None

        extension = Path(file_name).suffix.lower()

        if extension == ".pdf" and use_llm:
            text_content = "PDF processing logic here"  # Placeholder for PDF processing
        else:
            result = md.convert(tmp_path)
            text_content = result.text_content.strip() if result and result.text_content else ""

        return text_content
    except Exception as e:
        raise e
    finally:
        os.unlink(tmp_path)

def main():
    st.set_page_config(
        page_title="DocuInsight",
        page_icon="📄",
        layout="wide"
    )

    with st.sidebar:
        st.title("DocuInsight")
        st.write("Make sense of your stuff")
        
        uploaded_files = st.file_uploader(
            "Choose file(s)", 
            type=['pdf', 'pptx', 'docx', 'xlsx', 'jpg', 'png', 'html', 'csv', 'json', 'xml'], 
            accept_multiple_files=True
        )

        st.subheader("Settings")
        use_llm = st.checkbox("Use AI for Better Analysis", value=True)
        llm_provider = st.radio("Select LLM Provider", ["OpenAI","Local"], index=0)

        custom_api_key = None
        if use_llm and llm_provider == "OpenAI":
            st.info("OpenAI requires an API key for enhanced analysis")
            use_custom_key = st.checkbox("Use your own OpenAI API key")
            if use_custom_key:
                custom_api_key = st.text_input(
                    "Enter OpenAI API Key",
                    type="password",
                    help="Your API key will not be stored and is only used for your session."
                )

    if uploaded_files:
        if "extracted_texts" not in st.session_state:
            st.session_state.extracted_texts = {}

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.extracted_texts:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        text_content = process_document_cached(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                            use_llm,
                            llm_provider,
                            custom_api_key
                        )
                        st.session_state.extracted_texts[uploaded_file.name] = text_content
                    except Exception as e:
                        st.error(f"Error processing document {uploaded_file.name}: {str(e)}")
                        st.session_state.extracted_texts[uploaded_file.name] = f"[Error: {str(e)}]"

        extracted_texts = st.session_state.extracted_texts

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for file_name, content in extracted_texts.items():
                zf.writestr(f"{file_name}_extracted.txt", content)
        zip_buffer.seek(0)

        col_left, col_right = st.columns([0.8, 0.2])
        with col_right:
            st.download_button(
                "Download All Content",
                data=zip_buffer.getvalue(),
                file_name="all_extracted.zip",
                mime="application/zip"
            )

        tabs = st.tabs([uploaded_file.name for uploaded_file in uploaded_files])
        for index, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            text_content = extracted_texts[file_name]

            with tabs[index]:
                colA, colB = st.columns([0.8, 0.2])
                with colB:
                    st.download_button(
                        "Download Content",
                        text_content,
                        file_name=f"{file_name}_extracted.txt",
                        mime="text/plain"
                    )
                st.text_area("Content", text_content, height=600)

    else:
        st.info("Please upload some content to begin analysis")

if __name__ == "__main__":
    main()
