# DocuInsight

An interactive tool designed to process and analyse documents of various types, powered by Microsoft’s **MarkItDown** library. This application offers an optional AI-assisted feature for image descriptions via GPT-4o (if you provide an OpenAI API key) or use a local model.

## Key Features

- **Broad File Support**: PDF, PPTX, DOCX, XLSX, images, audio, and more  
- **AI-Enhanced Image Descriptions**: Integrate GPT-4o for deeper analysis (optional)  
- **User-Friendly Interface**: Built with Streamlit for straightforward build  
- **Export Options**: Save text outputs for offline reference  
- **Temporary File Management**: Uploaded files are processed securely and removed upon completion  

## Getting Started

### Prerequisites

- Python 3.12  
- (Optional) An OpenAI key for GPT-4o functionality

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ryanmcdonough/DocuInsight.git
cd DocuInsight
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
touch .env

# Add your OpenAI API key (optional)
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

or rename the `.env.example` to `.env` and update the key in there.

4. Run the application:
```bash
streamlit run main.py
```

## 💻 Usage

1. Launch the application
2. Upload your documents using the sidebar
3. Toggle AI enhancement if desired, this will default to OpenAI but you can use your own Local Model
4. View extracted content and document information in the respective tabs
5. Download the extracted content as needed

## 📋 Supported Formats

- PDF documents
- PowerPoint presentations (PPTX)
- Word documents (DOCX)
- Excel spreadsheets (XLSX)
- Images (JPG, PNG) with EXIF data and OCR
- (Coming Soon) Audio files (MP3, WAV) with EXIF data and transcription
- HTML files
- Text-based files (CSV, JSON, XML)

## ⚙️ Configuration

The application can be configured using environment variables or through the UI:

- `OPENAI_API_KEY`: Your OpenAI API key for AI enhancement
- Custom API key input available in the UI


## 📝 License & MS Repo

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Orginal MS Markitdown repo: https://github.com/microsoft/markitdown

## 🙏 Acknowledgments

- Microsoft MarkItDown technology
- Streamlit framework
