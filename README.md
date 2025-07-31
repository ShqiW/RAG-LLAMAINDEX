# LlamaIndex RAG Demo

A demonstration project for RAG (Retrieval-Augmented Generation) system built with LlamaIndex.

## Project Structure

```
LLamaIndex_RAG/
├── data/                          # Document data directory
│   ├── AIA_2024_firm_survey_report_Infographic.pdf
│   ├── Amazon-2024-Annual-Report.pdf
│   └── compact-guide-to-large-language-models.pdf
├── interactive_rag.py            # Interactive query script
├── test_api.py                   # API connection test script
├── requirements.txt              # Dependency package list
└── README.md                    # Project documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Set up your OpenAI API key:

```bash
# macOS/Linux
export OPENAI_API_KEY=your_api_key_here

# Windows
set OPENAI_API_KEY=your_api_key_here
```

Or set it directly in the code:
```python
os.environ['OPENAI_API_KEY'] = 'your_api_key_here'
```

### 3. Run the Project

**Interactive Query (Recommended)**
```bash
python interactive_rag.py
```

**API Connection Test**
```bash
python test_api.py
```

## Features

- **Document Processing**: Supports PDF, DOCX, PPTX formats with automatic chunking and vectorization
- **Semantic Search**: Intelligent retrieval based on similarity
- **Multi-turn Conversation**: Chat functionality with context memory
- **Debug Mode**: View retrieval results and similarity scores
- **Data Agents**: Tool chain integration and automatic reasoning

## Usage Examples

### Basic Query
```python
response = query_engine.query("What is Amazon's revenue in 2024?")
print(response)
```

### Chat Conversation
```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What was Amazon's revenue in 2024?")
```

### Debug Retrieval Results
```python
# Enter 'debug' in interactive mode to view retrieval content
```

## Interactive Commands

- Enter any question for querying
- `debug` - View retrieval results from last query
- `debug <query>` - View retrieval results for specified query
- `quit` or `exit` - Exit the program

## Example Q&A

**Q**: What is Amazon's revenue in 2024?

**A**: Amazon's revenue in 2024 was $637,959 million.

**Q**: In AIA firm, what's the figure of architecture firm billing in 2023?

**A**: The architecture firm gross billings in 2023 were $104.1 billion.

**Q**: What are organizations using large language models for?

**A**: Organizations use large language models for various purposes, including chatbots and virtual assistants for customer support and open-ended conversations, code generation and debugging by providing useful code snippets, sentiment analysis to gauge emotions and opinions in text, text classification and clustering to identify themes and trends, language translation to globalize content, summarization and paraphrasing of large texts, and content generation for brainstorming and drafting ideas.


