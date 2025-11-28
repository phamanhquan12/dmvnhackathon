import random
import time
import asyncio
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import app modules
from app.core.database import get_async_session
from app.models.user import User
from app.models.document import FileTypeEnum
from app.services.rag.ingestion import IngestionService
from app.services.rag.chat import ChatService


def run_async(coro):
    """Helper to run async code in Streamlit's sync context."""
    # Set Windows-compatible event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create a new event loop for each call to avoid loop conflicts
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def get_or_create_user(employee_id: str, full_name: str) -> User:
    """Get existing user or create a new one in the database."""
    session_factory = get_async_session()
    async with session_factory() as session:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            # Update name if changed
            if user.full_name != full_name:
                user.full_name = full_name
                await session.commit()
            return user
        
        # Create new user
        user = User(employee_id=employee_id, full_name=full_name)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def ingest_pdf_file(file_path: str, filename: str) -> Dict[str, Any]:
    """Process and ingest a PDF file into the database."""
    session_factory = get_async_session()
    async with session_factory() as session:
        service = IngestionService(session)
        result = await service.process_document(file_path, filename, FileTypeEnum.PDF)
        return result


async def get_all_documents() -> List[Dict[str, Any]]:
    """Fetch all documents from the database for UI selection."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        documents = await chat_service.get_all_documents()
        return [{"id": doc.id, "title": doc.title, "file_path": doc.file_path, "num_pages": doc.num_pages} for doc in documents]


async def rag_chat(query: str, document_ids: Optional[List[int]] = None) -> str:
    """Perform RAG-based chat using the ChatService with optional document filtering."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        response = await chat_service.chat(query, document_ids=document_ids)
        return response


async def generate_flashcards(document_id: Optional[int] = None, title: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate flashcards from selected document using the ChatService."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        flashcards = await chat_service.flash_cards(id=document_id, title=title)
        return flashcards


async def generate_quiz(document_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Generate quiz questions using the ChatService with optional document filtering."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        quiz_data = await chat_service.quiz(document_ids=document_ids)
        return quiz_data


async def grade_quiz(quiz_response: List[Dict], user_answers: Dict[int, str], user_id: str) -> Dict[str, Any]:
    """Grade the quiz and save results to TheorySession."""
    session_factory = get_async_session()
    async with session_factory() as session:
        possible_choices = ['A', 'B', 'C', 'D']
        score = 0
        wrong_questions_answers = {'question': [], 'user_answer': [], 'correct_answer': [], 'explanation': []}
        
        for i, q in enumerate(quiz_response):
            question = q.get("question")
            options = q.get("options", [])
            correct_answer = q.get("answer")
            explanation = q.get("explanation", "")
            
            user_answer = user_answers.get(i)
            if user_answer and user_answer in possible_choices:
                user_answer_index = possible_choices.index(user_answer)
                user_answer_text = options[user_answer_index] if user_answer_index < len(options) else ""
                
                if user_answer_text == correct_answer:
                    score += 1
                else:
                    wrong_questions_answers['question'].append(question)
                    wrong_questions_answers['user_answer'].append(user_answer_text)
                    wrong_questions_answers['correct_answer'].append(correct_answer)
                    wrong_questions_answers['explanation'].append(explanation)
        
        total = len(quiz_response)
        status = "PASSED" if total > 0 and (score / total) >= 0.6 else "FAILED"
        
        # Get user UUID from employee_id
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            from app.models.session import TheorySession
            session_obj = TheorySession(
                user_id=user.id,
                score=score,
                status=status,
                details=wrong_questions_answers
            )
            session.add(session_obj)
            await session.commit()
        
        return {
            "score": score,
            "total": total,
            "status": status,
            "details": wrong_questions_answers
        }


def init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = {
            "use_mocks": False,  # Default to using real RAG services
            "study_url": "",
            "vision_url": "",
            "chat_base": "",
            "chat_key": "",
            "chat_model": "gpt-3.5-turbo",
        }
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("quiz", None)
    st.session_state.setdefault("vision", None)
    st.session_state.setdefault("user_name", "")
    st.session_state.setdefault("employee_id", "")
    st.session_state.setdefault("user_db_id", None)  # Store user UUID from DB
    st.session_state.setdefault("machine", "")
    st.session_state.setdefault("mode", "Practice")
    st.session_state.setdefault(
        "flashcards",
        [
            {"front": "Pre-start safety", "back": "E-stop, guards closed, PPE, clear workspace."},
            {"front": "Torque sequence", "back": "Tighten bolts in a star pattern; 18 Nm final torque."},
            {"front": "Clamp verification", "back": "Check both side clamps engaged before cycle start."},
            {"front": "Sensor alignment", "back": "Verify ejector and part-present sensors with dry run."},
            {"front": "Shutdown steps", "back": "Stop conveyor, release pressure, lock out main power."},
        ],
    )
    st.session_state.setdefault("flash_index", 0)
    st.session_state.setdefault("flash_revealed", False)
    st.session_state.setdefault(
        "chat_history",
        [{"role": "system", "content": "You are a concise training assistant for factory SOPs."}],
    )
    st.session_state.setdefault("chat_reply", "")
    st.session_state.setdefault("rag_quiz_questions", [])  # RAG-generated quiz questions
    st.session_state.setdefault("rag_quiz_answers", {})    # User answers for RAG quiz
    st.session_state.setdefault("rag_quiz_result", None)   # RAG quiz grading result
    st.session_state.setdefault("test_questions", [])
    st.session_state.setdefault("test_answers", {})
    st.session_state.setdefault("test_result", None)
    st.session_state.setdefault("machine_steps", {
        "SOP-04 Press start": ["E-stop test", "Guards closed", "Dry run", "Auto start"],
        "SOP-09 Bolt fastening": ["Align fixture", "Clamp both sides", "Torque 18 Nm", "Visual check"],
        "SOP-12 Conveyor clear": ["Check belts", "Remove debris", "Test jog", "Set speed"],
    })
    st.session_state.setdefault("ingestion_status", None)  # Track PDF ingestion status
    st.session_state.setdefault("available_documents", [])  # List of documents from DB
    st.session_state.setdefault("selected_doc_ids", [])     # Selected document IDs for filtering
    st.session_state.setdefault("generated_flashcards", []) # AI-generated flashcards
    st.session_state.setdefault("gen_flash_index", 0)       # Index for generated flashcards
    st.session_state.setdefault("gen_flash_revealed", False)  # Reveal state for generated flashcards


def add_event(evt_type: str, message: str) -> None:
    st.session_state.events.insert(0, {"type": evt_type, "message": message, "at": datetime.now()})
    st.session_state.events = st.session_state.events[:30]


def sleep_random(base: int, jitter: int) -> None:
    time.sleep((base + random.randint(0, jitter)) / 1000)


def mock_study_ask(question: str) -> Dict[str, Any]:
    sleep_random(450, 400)
    sample = [
        "1) E-stop and guards closed; 2) Clear workspace; 3) Air/power in range; 4) Dry-run in manual; 5) PPE on.",
        "Check oil level, clean the die, confirm sensor alignment, then run a single cycle in jog mode to validate ejectors.",
    ]
    return {
        "answer": random.choice(sample),
        "sources": ["SOP-04 Section 2.1", "Trainer note v3", "Safety bulletin #18"],
        "confidence": 0.82,
        "followUps": ["Show the visual checklist.", "What is a critical fault vs. soft fault?"],
    }


def mock_quiz(module: str, count: int, language: str) -> Dict[str, Any]:
    sleep_random(520, 420)
    questions = []
    for i in range(count):
        questions.append(
            {
                "id": i + 1,
                "text": f"({language.upper()}) What is step {i + 1} in {module}?",
                "expected": ["Secure workpiece", "Check torque settings", "Run dry test"][i % 3],
                "difficulty": ["Easy", "Medium", "Hard"][i % 3],
            }
        )
    return {"module": module, "language": language, "questions": questions}


def mock_vision(scenario: str, operator: str, filename: str) -> Dict[str, Any]:
    sleep_random(700, 400)
    violations = [
        "Skipped tightening sequence on bolt 3",
        "Hand inside guard area while conveyor active",
        "Torque wrench not reset to 18 Nm",
    ]
    timeline = [
        {"t": "00:04", "event": "Detected operator hands and tool"},
        {"t": "00:11", "event": "Step 2 completed: align fixture"},
        {"t": "00:22", "event": "Flagged deviation: missing clamp on side B"},
        {"t": "00:34", "event": "Score computed and report packaged"},
    ]
    return {
        "scenario": scenario,
        "operator": operator,
        "score": f"{75 + random.random() * 15:.1f}",
        "passed": random.random() > 0.3,
        "keyFindings": violations[:2],
        "timeline": timeline,
        "fileName": filename,
    }


def mock_chat(user_message: str) -> str:
    sleep_random(300, 200)
    examples = [
        "Remember the five-point safety check before every cycle. Want me to list it?",
        "Focus on clamp verification; it prevents most part shifts.",
        "If the torque wrench is above 20 Nm, recalibrate before use.",
    ]
    return random.choice(examples)


def build_test_questions() -> List[Dict[str, Any]]:
    base = [
        ("When must the E-stop be tested?", ["Start of shift", "End of shift", "Never"], "Start of shift"),
        ("What is the target torque for SOP-09?", ["12 Nm", "18 Nm", "24 Nm"], "18 Nm"),
        ("What PPE is mandatory?", ["Gloves and glasses", "Helmet only", "None"], "Gloves and glasses"),
        ("What to do before auto mode?", ["Dry-run in manual", "Skip manual", "Full speed immediately"], "Dry-run in manual"),
        ("Where to check clamp status?", ["HMI clamp screen", "Torque wrench", "Oil gauge"], "HMI clamp screen"),
        ("How to log a defect?", ["In the MES app", "Write on paper only", "Do nothing"], "In the MES app"),
        ("What triggers a soft fault?", ["Minor order swap", "Guard open", "Motor overheating"], "Minor order swap"),
        ("How many retries allowed?", ["1", "3", "Unlimited"], "3"),
        ("Who clears an E-stop?", ["Trainer or coach", "Operator alone", "Maintenance only"], "Trainer or coach"),
        ("What to inspect after jam?", ["Sensor alignment", "Change SOP", "Increase speed"], "Sensor alignment"),
    ]
    return [
        {"id": idx + 1, "question": q, "options": opts, "answer": ans}
        for idx, (q, opts, ans) in enumerate(base)
    ]


def study_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if st.session_state.config["use_mocks"] or not st.session_state.config["study_url"]:
        if endpoint == "/ask":
            return mock_study_ask(payload["question"])
        if endpoint == "/quiz":
            return mock_quiz(payload["module"], payload["count"], payload["language"])
    res = requests.post(
        st.session_state.config["study_url"].rstrip("/") + endpoint,
        json=payload,
        timeout=30,
    )
    res.raise_for_status()
    return res.json()


def vision_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if st.session_state.config["use_mocks"] or not st.session_state.config["vision_url"]:
        filename = payload["file"].name if payload.get("file") else "mock-video.mp4"
        return mock_vision(payload["scenario"], payload["operator"], filename)

    files = {"file": payload["file"]} if payload.get("file") else None
    data = {"scenario": payload["scenario"], "operator": payload["operator"]}
    res = requests.post(
        st.session_state.config["vision_url"].rstrip("/") + endpoint,
        data=data,
        files=files,
        timeout=60,
    )
    res.raise_for_status()
    return res.json()


def chat_request(user_message: str) -> str:
    base = st.session_state.config["chat_base"].rstrip("/") if st.session_state.config["chat_base"] else ""
    key = st.session_state.config["chat_key"]
    model = st.session_state.config.get("chat_model") or "gpt-3.5-turbo"

    if st.session_state.config["use_mocks"] or not base or not key:
        return mock_chat(user_message)

    messages = st.session_state.chat_history + [{"role": "user", "content": user_message}]
    res = requests.post(
        base + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.4},
        timeout=45,
    )
    res.raise_for_status()
    data = res.json()
    choice = data.get("choices", [{}])[0]
    return choice.get("message", {}).get("content", "")


def render_events() -> None:
    st.subheader("Events")
    if not st.session_state.events:
        st.info("No events yet. Try sending a request.")
        return
    badges = {
        "ask": "[ask]",
        "quiz": "[quiz]",
        "vision": "[vision]",
        "flash": "[flash]",
        "chat": "[chat]",
        "test": "[test]",
    }
    for evt in st.session_state.events:
        badge = badges.get(evt["type"], "[info]")
        st.write(f"{badge} {evt['message']} - {evt['at'].strftime('%H:%M:%S')}")


def layout() -> None:
    st.set_page_config(page_title="DENSO-MIND Demo Console", layout="wide")
    st.title("DENSO-MIND - Dual backend demo")
    st.caption("Practice (chat + flashcards) | Testing (quiz + live video) | Admin settings")

    with st.sidebar:
        st.markdown("### Navigation")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        if col_nav1.button("Practice", use_container_width=True):
            st.session_state.mode = "Practice"
        if col_nav2.button("Testing", use_container_width=True):
            st.session_state.mode = "Testing"
        if col_nav3.button("Settings", use_container_width=True):
            st.session_state.mode = "Settings"
        st.markdown("---")
        st.markdown("### Profile")
        st.write(f"Name: {st.session_state.user_name or 'Not set'}")
        st.write(f"ID: {st.session_state.employee_id or 'Not set'}")
        st.write(f"Machine: {st.session_state.machine or 'Not selected'}")
        if st.button("Switch machine (quick)"):
            st.session_state.machine = st.selectbox(
                "Pick machine",
                list(st.session_state.machine_steps.keys()),
                key="sidebar_switch_machine",
                index=0,
            )
            add_event("info", f"Switched machine to {st.session_state.machine}")
        if st.button("Switch account (reset name/ID)"):
            st.session_state.user_name = ""
            st.session_state.employee_id = ""
            st.rerun()

    if not st.session_state.user_name or not st.session_state.employee_id:
        st.markdown("### Trainee sign-in")
        with st.form("profile_form"):
            name = st.text_input("Name", st.session_state.user_name)
            emp_id = st.text_input("Employee ID", st.session_state.employee_id)
            machine = st.selectbox("Machine/SOP", list(st.session_state.machine_steps.keys()), index=0)
            saved = st.form_submit_button("Sign in")
        if saved and name.strip() and emp_id.strip():
            st.session_state.user_name = name.strip()
            st.session_state.employee_id = emp_id.strip()
            st.session_state.machine = machine
            # Save user to database
            try:
                user = run_async(get_or_create_user(emp_id.strip(), name.strip()))
                st.session_state.user_db_id = str(user.id)
                add_event("info", f"User '{name.strip()}' signed in and saved to database.")
            except Exception as e:
                add_event("info", f"Profile saved (DB error: {e}).")
            st.rerun()
        elif saved:
            st.warning("Please fill in both Name and Employee ID.")
        st.stop()

    top_cols = st.columns([2, 1])
    with top_cols[0]:
        st.write(f"User: **{st.session_state.user_name}** | ID: **{st.session_state.employee_id}**")
        st.write(f"Machine: **{st.session_state.machine or 'Not selected'}**")
    with top_cols[1]:
        st.write(f"Mode: **{st.session_state.mode}**")

    if st.session_state.mode == "Practice":
        st.subheader("Practice")
        practice_tabs = st.tabs(["Chatbot + Steps", "Flashcards (AI Generated)", "Upload Documents", "View Documents"])
        
        with practice_tabs[0]:
            st.markdown(f"Current machine: **{st.session_state.machine or 'Not selected'}**")
            steps = st.session_state.machine_steps.get(
                st.session_state.machine or list(st.session_state.machine_steps.keys())[0], []
            )
            st.markdown("Steps / summary")
            for s in steps:
                st.write("-", s)
            
            st.markdown("---")
            st.markdown("### RAG-Powered Chat")
            st.caption("Ask questions about uploaded documents (uses AI to search and answer)")
            
            # Document selection for filtering
            st.markdown("**Select documents to query (optional):**")
            if st.button("🔄 Refresh document list", key="refresh_docs_chat"):
                try:
                    docs = run_async(get_all_documents())
                    st.session_state.available_documents = docs
                    add_event("info", f"Loaded {len(docs)} documents.")
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
            
            if st.session_state.available_documents:
                doc_options = {f"{doc['title']} (ID: {doc['id']}, {doc['num_pages']} pages)": doc['id'] 
                              for doc in st.session_state.available_documents}
                selected_docs = st.multiselect(
                    "Filter by documents:",
                    options=list(doc_options.keys()),
                    default=[],
                    key="doc_filter_chat"
                )
                st.session_state.selected_doc_ids = [doc_options[d] for d in selected_docs] if selected_docs else []
            else:
                st.info("Click 'Refresh document list' to load available documents.")
            
            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("Ask a question about the documents", placeholder="e.g., What are the safety procedures?")
                chat_submit = st.form_submit_button("Send")
            
            if chat_submit and user_msg.strip():
                try:
                    with st.spinner("Searching documents and generating response..."):
                        doc_ids = st.session_state.selected_doc_ids if st.session_state.selected_doc_ids else None
                        reply = run_async(rag_chat(user_msg.strip(), document_ids=doc_ids))
                    st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.session_state.chat_reply = reply
                    add_event("chat", "RAG Chatbot responded.")
                except Exception as exc:
                    st.error(f"RAG Chat error: {exc}")
                    add_event("chat", f"Error: {exc}")
            
            if st.session_state.chat_reply:
                st.success(st.session_state.chat_reply)
            
            st.markdown("**Recent conversation:**")
            for msg in st.session_state.chat_history[-6:]:
                if msg["role"] == "system":
                    continue
                role = "You" if msg["role"] == "user" else "Bot"
                st.markdown(f"*{role}:* {msg['content']}")

        with practice_tabs[1]:
            st.subheader("Flashcards (AI Generated)")
            st.caption("Generate flashcards from uploaded training documents using AI.")
            
            # Document selection for flashcard generation
            st.markdown("**Select a document to generate flashcards:**")
            if st.button("🔄 Refresh document list", key="refresh_docs_flash"):
                try:
                    docs = run_async(get_all_documents())
                    st.session_state.available_documents = docs
                    add_event("info", f"Loaded {len(docs)} documents.")
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
            
            if st.session_state.available_documents:
                doc_options_flash = {f"{doc['title']} (ID: {doc['id']})": doc['id'] 
                                    for doc in st.session_state.available_documents}
                selected_doc_flash = st.selectbox(
                    "Choose a document:",
                    options=["-- Select a document --"] + list(doc_options_flash.keys()),
                    key="doc_select_flash"
                )
                
                if st.button("Generate Flashcards", type="primary"):
                    if selected_doc_flash and selected_doc_flash != "-- Select a document --":
                        with st.spinner("Generating flashcards from document..."):
                            try:
                                doc_id = doc_options_flash[selected_doc_flash]
                                flashcards = run_async(generate_flashcards(document_id=doc_id))
                                if isinstance(flashcards, list) and len(flashcards) > 0:
                                    st.session_state.generated_flashcards = flashcards
                                    st.session_state.gen_flash_index = 0
                                    st.session_state.gen_flash_revealed = False
                                    add_event("flash", f"Generated {len(flashcards)} flashcards.")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate flashcards. Try again.")
                            except Exception as e:
                                st.error(f"Error generating flashcards: {e}")
                    else:
                        st.warning("Please select a document first.")
            else:
                st.info("Click 'Refresh document list' to load available documents.")
            
            st.markdown("---")
            
            # Display flashcards
            if st.session_state.generated_flashcards:
                cards = st.session_state.generated_flashcards
                idx = st.session_state.gen_flash_index
                card = cards[idx]
                
                st.write(f"**Card {idx + 1} of {len(cards)}**")
                
                # Display card in a nice format
                st.markdown(f"### ❓ {card.get('question', card.get('front', 'N/A'))}")
                
                if st.session_state.gen_flash_revealed:
                    st.success(f"**Answer:** {card.get('answer', card.get('back', 'N/A'))}")
                
                cols = st.columns(3)
                if cols[0].button("⬅️ Prev", key="flash_prev"):
                    st.session_state.gen_flash_index = (idx - 1) % len(cards)
                    st.session_state.gen_flash_revealed = False
                    add_event("flash", "Moved to previous flashcard.")
                    st.rerun()
                if cols[1].button("👁️ Reveal/Hide", key="flash_reveal"):
                    st.session_state.gen_flash_revealed = not st.session_state.gen_flash_revealed
                    st.rerun()
                if cols[2].button("➡️ Next", key="flash_next"):
                    st.session_state.gen_flash_index = (idx + 1) % len(cards)
                    st.session_state.gen_flash_revealed = False
                    add_event("flash", "Moved to next flashcard.")
                    st.rerun()
                
                # Progress bar
                st.progress((idx + 1) / len(cards))
            else:
                st.info("No flashcards yet. Select a document and click 'Generate Flashcards' to create them.")
        
        with practice_tabs[2]:
            st.subheader("Upload Training Documents (PDF)")
            st.caption("Upload PDF documents to add them to the knowledge base for RAG chat and quiz generation.")
            
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
            
            if uploaded_file is not None:
                st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
                
                if st.button("Process and Ingest Document"):
                    with st.spinner("Processing PDF and generating embeddings..."):
                        try:
                            # Save uploaded file to temp location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Run ingestion
                            result = run_async(ingest_pdf_file(tmp_path, uploaded_file.name))
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                            st.session_state.ingestion_status = result
                            add_event("info", f"Ingested document: {uploaded_file.name} ({result['num_chunks']} chunks)")
                            st.success(f"Document '{result['title']}' ingested successfully! Created {result['num_chunks']} chunks.")
                        except Exception as e:
                            st.error(f"Ingestion error: {e}")
                            add_event("info", f"Ingestion failed: {e}")
            
            if st.session_state.ingestion_status:
                st.markdown("**Last ingestion result:**")
                st.json(st.session_state.ingestion_status)
        
        with practice_tabs[3]:
            st.subheader("View Documents")
            st.caption("Browse and view uploaded training documents.")
            
            if st.button("🔄 Refresh document list", key="refresh_docs_view"):
                try:
                    docs = run_async(get_all_documents())
                    st.session_state.available_documents = docs
                    add_event("info", f"Loaded {len(docs)} documents.")
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
            
            if st.session_state.available_documents:
                st.markdown("### Available Documents")
                for doc in st.session_state.available_documents:
                    with st.expander(f"📄 {doc['title']} ({doc['num_pages']} pages)"):
                        st.write(f"**Document ID:** {doc['id']}")
                        st.write(f"**Pages:** {doc['num_pages']}")
                        st.write(f"**File Path:** {doc['file_path']}")
                        
                        # PDF viewing (if file exists locally)
                        if doc['file_path'] and os.path.exists(doc['file_path']):
                            try:
                                with open(doc['file_path'], "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=pdf_bytes,
                                        file_name=f"{doc['title']}.pdf",
                                        mime="application/pdf",
                                        key=f"download_{doc['id']}"
                                    )
                            except Exception as e:
                                st.warning(f"Could not load file: {e}")
                        else:
                            st.info("PDF file not available for viewing (stored in temp location during ingestion).")
            else:
                st.info("No documents available. Click 'Refresh document list' or upload documents first.")

    if st.session_state.mode == "Testing":
        st.subheader("Testing")
        test_tabs = st.tabs(["RAG Quiz (AI Generated)", "Static Quiz", "Live video demo"])
        
        # RAG Quiz Tab - Generated from uploaded documents
        with test_tabs[0]:
            st.markdown("### AI-Generated Quiz from Documents")
            st.caption("Quiz questions are generated from the ingested training documents using AI.")
            
            # Document selection for quiz generation
            st.markdown("**Select documents for quiz generation (optional):**")
            if st.button("🔄 Refresh document list", key="refresh_docs_quiz"):
                try:
                    docs = run_async(get_all_documents())
                    st.session_state.available_documents = docs
                    add_event("info", f"Loaded {len(docs)} documents.")
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
            
            quiz_doc_ids = None
            if st.session_state.available_documents:
                doc_options_quiz = {f"{doc['title']} (ID: {doc['id']}, {doc['num_pages']} pages)": doc['id'] 
                                   for doc in st.session_state.available_documents}
                selected_docs_quiz = st.multiselect(
                    "Filter quiz by documents:",
                    options=list(doc_options_quiz.keys()),
                    default=[],
                    key="doc_filter_quiz"
                )
                quiz_doc_ids = [doc_options_quiz[d] for d in selected_docs_quiz] if selected_docs_quiz else None
            else:
                st.info("Click 'Refresh document list' to load available documents, or generate quiz from all documents.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate New Quiz", type="primary"):
                    with st.spinner("Generating quiz questions from documents..."):
                        try:
                            quiz_data = run_async(generate_quiz(document_ids=quiz_doc_ids))
                            if isinstance(quiz_data, list):
                                st.session_state.rag_quiz_questions = quiz_data
                                st.session_state.rag_quiz_answers = {}
                                st.session_state.rag_quiz_result = None
                                add_event("quiz", f"Generated {len(quiz_data)} quiz questions.")
                                st.rerun()
                            else:
                                st.error("Failed to generate quiz. Make sure you have uploaded documents first.")
                        except Exception as e:
                            st.error(f"Quiz generation error: {e}")
            with col2:
                if st.button("Clear Quiz"):
                    st.session_state.rag_quiz_questions = []
                    st.session_state.rag_quiz_answers = {}
                    st.session_state.rag_quiz_result = None
                    st.rerun()
            
            if st.session_state.rag_quiz_questions:
                st.markdown("---")
                possible_choices = ['A', 'B', 'C', 'D']
                
                for i, q in enumerate(st.session_state.rag_quiz_questions):
                    st.markdown(f"**Câu {i + 1}:** {q.get('question', 'N/A')}")
                    options = q.get("options", [])
                    
                    # Create radio options
                    selected = st.radio(
                        "Chọn đáp án:",
                        options=possible_choices[:len(options)],
                        format_func=lambda x, opts=options, choices=possible_choices: opts[choices.index(x)] if choices.index(x) < len(opts) else x,
                        key=f"rag_quiz_{i}",
                        index=None,
                        horizontal=True
                    )
                    if selected:
                        st.session_state.rag_quiz_answers[i] = selected
                    st.markdown("---")
                
                if st.button("Submit Quiz & Get Score", type="primary"):
                    if len(st.session_state.rag_quiz_answers) < len(st.session_state.rag_quiz_questions):
                        st.warning("Please answer all questions before submitting.")
                    else:
                        with st.spinner("Grading quiz and saving results..."):
                            try:
                                result = run_async(grade_quiz(
                                    st.session_state.rag_quiz_questions,
                                    st.session_state.rag_quiz_answers,
                                    st.session_state.employee_id
                                ))
                                st.session_state.rag_quiz_result = result
                                add_event("quiz", f"Quiz graded: {result['score']}/{result['total']} - {result['status']}")
                            except Exception as e:
                                st.error(f"Grading error: {e}")
                
                if st.session_state.rag_quiz_result:
                    result = st.session_state.rag_quiz_result
                    if result["status"] == "PASSED":
                        st.success(f"🎉 PASSED: {result['score']}/{result['total']} correct (60% needed to pass)")
                    else:
                        st.error(f"❌ FAILED: {result['score']}/{result['total']} correct (60% needed to pass)")
                    
                    if result["details"]["question"]:
                        st.markdown("**Incorrect Answers:**")
                        for idx, (q, ua, ca, exp) in enumerate(zip(
                            result["details"]["question"],
                            result["details"]["user_answer"],
                            result["details"]["correct_answer"],
                            result["details"]["explanation"]
                        )):
                            with st.expander(f"Question: {q[:50]}..."):
                                st.write(f"**Your answer:** {ua}")
                                st.write(f"**Correct answer:** {ca}")
                                st.write(f"**Explanation:** {exp}")
            else:
                st.info("Click 'Generate New Quiz' to create questions from uploaded documents.")
        
        # Static Quiz Tab
        with test_tabs[1]:
            st.markdown("### Static Pre-defined Quiz")
            if not st.session_state.test_questions:
                st.session_state.test_questions = build_test_questions()
            if st.button("Regenerate test"):
                st.session_state.test_questions = build_test_questions()
                st.session_state.test_answers = {}
                st.session_state.test_result = None
            for q in st.session_state.test_questions:
                st.markdown(f"**#{q['id']}** {q['question']}")
                choice = st.radio("Pick one", q["options"], key=f"test_{q['id']}", index=None)
                if choice:
                    st.session_state.test_answers[q["id"]] = choice
                st.markdown("---")
            if st.button("Grade test"):
                correct = 0
                for q in st.session_state.test_questions:
                    if st.session_state.test_answers.get(q["id"]) == q["answer"]:
                        correct += 1
                passed = correct >= 7
                st.session_state.test_result = {"score": correct, "passed": passed}
                add_event("test", f"Test graded: {correct}/10 {'Pass' if passed else 'Fail'}")
            if st.session_state.test_result:
                res = st.session_state.test_result
                status = "Pass" if res["passed"] else "Fail"
                st.success(f"{status}: {res['score']}/10 correct (need 7 to pass)")

        with test_tabs[2]:
            st.markdown("SOP and live feed (demo)")
            steps = st.session_state.machine_steps.get(
                st.session_state.machine or list(st.session_state.machine_steps.keys())[0], []
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**SOP steps**")
                for s in steps:
                    st.write("-", s)
            with col_b:
                st.markdown("**Live feed with AI overlay (placeholder)**")
                st.image(
                    "https://placehold.co/600x340/0b1224/ffffff?text=Live+Video+with+Overlay",
                    caption="Model overlay preview",
                    use_column_width=True,
                )
            st.markdown("Upload a clip for scoring")
            with st.form("vision_form"):
                scenario = st.text_input("SOP / Scenario", st.session_state.machine or "Bolt fastening / SOP-09")
                operator = st.text_input("Operator", st.session_state.user_name or "Operator")
                video = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
                vision_submit = st.form_submit_button("Score this attempt")
            if vision_submit and scenario.strip() and operator.strip() and video:
                try:
                    st.session_state.vision = vision_request(
                        "/score", {"scenario": scenario.strip(), "operator": operator.strip(), "file": video}
                    )
                    add_event("vision", f"Scored attempt for {operator.strip()} ({scenario.strip()}).")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error from vision API: {exc}")

            if st.session_state.vision:
                v = st.session_state.vision
                st.success(f"Score: {v['score']} / 100 - {'Pass' if v['passed'] else 'Needs retry'}")
                st.markdown("Key findings")
                for item in v.get("keyFindings", []):
                    st.write("-", item)
                st.markdown("Timeline")
                for t in v.get("timeline", []):
                    st.caption(f"{t['t']} - {t['event']}")
                st.caption(f"File: {v.get('fileName', 'N/A')}")

    if st.session_state.mode == "Settings":
        st.subheader("Settings (admin)")
        with st.form("settings_form"):
            use_mocks = st.checkbox("Use mock data (no backend needed)", value=st.session_state.config["use_mocks"])
            study_url = st.text_input("Study/Quiz base URL", st.session_state.config["study_url"])
            vision_url = st.text_input("Vision/Scoring base URL", st.session_state.config["vision_url"])
            st.markdown("Chat (OpenAI compatible)")
            chat_base = st.text_input("Chat base URL", st.session_state.config["chat_base"], placeholder="https://api.openai.com")
            chat_model = st.text_input("Chat model", st.session_state.config["chat_model"])
            chat_key = st.text_input("Chat API key", st.session_state.config["chat_key"], type="password")
            submitted = st.form_submit_button("Save settings")
        if submitted:
            st.session_state.config["use_mocks"] = use_mocks
            st.session_state.config["study_url"] = study_url.strip()
            st.session_state.config["vision_url"] = vision_url.strip()
            st.session_state.config["chat_base"] = chat_base.strip()
            st.session_state.config["chat_model"] = chat_model.strip() or "gpt-3.5-turbo"
            st.session_state.config["chat_key"] = chat_key.strip()
            add_event("info", f"Config saved. Mode: {'mock' if use_mocks else 'live'}")
        st.markdown("Current status")
        st.write(f"Study/Quiz: {'Mock' if st.session_state.config['use_mocks'] else study_url or 'Not set'}")
        st.write(f"Vision/Scoring: {'Mock' if st.session_state.config['use_mocks'] else vision_url or 'Not set'}")
        st.write(
            f"Chat: {'Mock' if st.session_state.config['use_mocks'] or not chat_base or not chat_key else 'Live'}"
        )

    st.markdown("---")
    render_events()


def main() -> None:
    init_state()
    layout()


if __name__ == "__main__":
    main()
