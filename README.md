# DENSO-MIND Backend

An AI-powered training and assessment platform for factory workers, featuring RAG (Retrieval-Augmented Generation) capabilities for interactive learning from SOP documents.

## 🎯 Features

### Stage A: Theory Session
- **Document Upload & Ingestion**: Upload PDF training documents that get processed, chunked, and vectorized for semantic search
- **RAG-Powered Chat**: Ask questions about uploaded documents with AI-generated answers based on relevant content
- **AI Flashcards**: Generate study flashcards from training materials
- **AI Quiz Generation**: Create multiple-choice quizzes from document content with automatic grading
- **User Management**: Track employee progress and save quiz results

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python with async SQLAlchemy
- **Database**: PostgreSQL with pgvector extension
- **AI/ML**: 
  - Google Gemini 2.5 Flash for text generation
  - Sentence Transformers for document embeddings
- **Vector Search**: pgvector for semantic similarity search

## 📁 Project Structure

```
DensoMind/
├── docker-compose.yml      # Docker services configuration (root level)
├── .env.example            # Environment variables template
├── README.md               # This file
└── backend/
    ├── main.py             # Streamlit application entry point
    ├── Dockerfile          # Application container
    ├── requirements-A.txt  # Python dependencies
    ├── app/
    │   ├── core/
    │   │   ├── config.py   # Application settings
    │   │   └── database.py # Database connection setup
    │   ├── models/
    │   │   ├── user.py     # User model
    │   │   ├── document.py # Document model
    │   │   ├── chunk.py    # Document chunk model with embeddings
    │   │   └── session.py  # Theory/Practical session models
    │   └── services/
    │       └── rag/
    │           ├── chat.py     # RAG chat, quiz, flashcard services
    │           ├── ingestion.py # Document processing pipeline
    │           ├── loaders.py  # PDF document loaders
    │           ├── splitters.py # Text chunking
    │           ├── vectorizer.py # Embedding generation
    │           └── utils.py    # Prompt templates
    └── alembic/            # Database migrations
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Google Gemini API key

### Option 1: Docker Deployment (Recommended)

1. **Clone and setup environment**
   ```bash
   # From project root
   cp .env.example .env
   ```

2. **Edit `.env` with your values**
   ```env
   POSTGRES_USER=denso_user
   POSTGRES_PASSWORD=your_secure_password
   POSTGRES_DB=denso_mind
   POSTGRES_HOST=db
   POSTGRES_PORT=5432
   API_KEY=your_google_gemini_api_key
   ```

3. **Build and run** (from project root)
   ```bash
   docker-compose up --build -d
   ```

4. **Access the application**
   - Open http://localhost:8501 in your browser

### Option 2: Local Development

1. **Start PostgreSQL with pgvector** (from project root)
   ```bash
   docker-compose up db -d
   ```

2. **Create virtual environment**
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-A.txt
   pip install streamlit google-generativeai pydantic-settings python-dotenv pypdf
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your values (use localhost for POSTGRES_HOST)
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the application**
   ```bash
   streamlit run main.py
   ```

## 📖 Usage Guide

### 1. Sign In
- Enter your name and employee ID
- Select your machine/SOP

### 2. Practice Mode

#### Chatbot + Steps
- Select documents to query (optional filter)
- Ask questions about the training materials
- Get AI-generated answers with context from documents

#### Flashcards (AI Generated)
- Select a document
- Click "Generate Flashcards"
- Study with interactive card navigation

#### Upload Documents
- Upload PDF training documents
- Documents are automatically processed and indexed

#### View Documents
- Browse uploaded documents
- Download PDFs

### 3. Testing Mode

#### RAG Quiz
- Select documents to generate quiz from (optional)
- Click "Generate New Quiz"
- Answer multiple-choice questions
- Submit for automatic grading
- Results are saved to your profile

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | PostgreSQL username | - |
| `POSTGRES_PASSWORD` | PostgreSQL password | - |
| `POSTGRES_DB` | Database name | - |
| `POSTGRES_HOST` | Database host | `localhost` |
| `POSTGRES_PORT` | Database port | `5432/5433` |
| `API_KEY` | Google Gemini API key | - |

## 🗄️ Database Schema

- **users**: Employee information (id, employee_id, full_name)
- **documents**: Uploaded training documents
- **document_chunks**: Chunked text with vector embeddings
- **theory_sessions**: Quiz results and progress tracking
- **practical_sessions**: Hands-on assessment records

## 📝 API Services

### ChatService
- `chat(query, document_ids)`: RAG-powered Q&A
- `quiz(document_ids)`: Generate quiz questions
- `flash_cards(id, title)`: Generate flashcards
- `get_all_documents()`: List available documents

### IngestionService
- `process_document(file_path, filename, file_type)`: Ingest and vectorize documents

## 🐳 Docker Commands

```bash
# Start all services (from project root)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build -d

# Access database
docker-compose exec db psql -U denso_user -d denso_mind

# View Docker volumes
docker volume ls | grep denso

# Backup database
docker-compose exec db pg_dump -U denso_user denso_mind > backup.sql
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is proprietary software developed for DENSO Vietnam.

## 👥 Team

DENSO Vietnam Hackathon Team
