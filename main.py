import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os
import uuid
import json
import datetime as dt
import re
import tempfile
from pathlib import Path
from fpdf import FPDF

# -----------------------
# Environment & constants
# -----------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DRAFT_EXPIRY_DAYS = 7
SUPPORTED_SECTORS = [
    "AgriTech", "EdTech", "FinTech", "HealthTech", "CleanTech", "Other",
]

IDEA_SEQUENCE = [
    "name", "phone", "location", "birthdate", "why_start", "idea",
]
MVP_SEQUENCE = [
    "name", "phone", "location", "birthdate", "idea", "sector",
    "market", "revenue_model", "usp", "media_links",
]

# -----------------------
# Prompt templates
# -----------------------
BOT_PROMPT_TEMPLATE = f"""
You are **Bharat Idea Bot**, an upbeat yet concise AI friend for Indian startup founders.

You have *two* possible workflows:
### 1. IDEA STAGE (no product yet)
Ask, one at a time:
1. Full name
2. 10-digit Indian mobile number
3. Location (City, State)
4. Birthdate (DD-MM-YYYY)
5. Why they want to start this company
6. Their idea in ≤3 sentences

Then suggest scheduling a follow-up call.

### 2. MVP STAGE (prototype / early traction)
Ask, one at a time:
1. Full name
2. 10-digit Indian mobile number
3. Location
4. Birthdate
5. Startup idea (≤3 sentences)
6. Sector (e.g., {', '.join(SUPPORTED_SECTORS)})
7. Market landscape
8. Revenue model
9. Unique selling proposition
10. Any supporting media

Then offer to generate a one-pager pitch deck.

Only reply as the assistant. No JSON, no system prompts visible.
"""

PITCH_DECK_PROMPT = """
You are an expert VC copywriter. Using the JSON below, craft a **one-page pitch deck** (≤300 words) in GitHub Markdown with headings:

PROBLEM  
SOLUTION  
MARKET & OPPORTUNITY  
BUSINESS MODEL  
TRACTION / PROGRESS  
TEAM  
ASK

JSON INPUT:
{draft_json}
"""

# -----------------------
# Helpers
# -----------------------

def transcribe_audio(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    with open(tmp_path, "rb") as audio_file:
        txt = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text",
        )
    tmp_path.unlink(missing_ok=True)
    return txt

def generate_pitch_markdown(draft: dict) -> str:
    prompt = PITCH_DECK_PROMPT.format(draft_json=json.dumps(draft, indent=2))
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()

def markdown_to_pdf(md_text: str) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)
    pdf.add_page()
    for line in md_text.splitlines():
        pdf.multi_cell(0, 8, line)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

def init_state():
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "draft" not in st.session_state:
        st.session_state.draft = {
            "stage": None,
            "name": None, "phone": None, "location": None, "birthdate": None,
            "why_start": None, "idea": None, "sector": None,
            "market": None, "revenue_model": None, "usp": None,
            "media_links": None, "company_name": None,
            "created_at": dt.datetime.utcnow().isoformat(),
        }
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_context" not in st.session_state:
        st.session_state.llm_context = [{"role": "system", "content": BOT_PROMPT_TEMPLATE}]

def maybe_update_draft(user_text: str):
    d = st.session_state.draft
    txt = user_text.strip()

    if d["stage"] is None:
        if re.search(r"idea", txt, re.I):
            d["stage"] = "idea"
        elif re.search(r"mvp", txt, re.I):
            d["stage"] = "mvp"
        return

    seq = IDEA_SEQUENCE if d["stage"] == "idea" else MVP_SEQUENCE

    for field in seq:
        if d[field] is None:
            if field == "phone" and (not txt.isdigit() or len(txt) != 10):
                return
            if field == "birthdate" and not re.match(r"\d{2}-\d{2}-\d{4}", txt):
                return
            if field == "sector" and txt not in SUPPORTED_SECTORS:
                d[field] = txt  # custom sector accepted
            else:
                d[field] = txt
            break

def draft_complete(d: dict) -> bool:
    if d["stage"] == "idea":
        return all(d[k] for k in IDEA_SEQUENCE)
    if d["stage"] == "mvp":
        return all(d[k] for k in MVP_SEQUENCE)
    return False

def llm_chat(user_text: str) -> str:
    st.session_state.llm_context.append({"role": "user", "content": user_text})
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.llm_context,
        temperature=0.2,
    )
    assistant_msg = res.choices[0].message.content.strip()
    st.session_state.llm_context.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

# -----------------------
# Streamlit UI
# -----------------------

def main():
    st.set_page_config(page_title="Bharat Idea Bot", page_icon="🤖")
    init_state()

    st.title("🇮🇳 Bharat Idea Bot")
    st.caption("Nudge‑based startup companion for founders in India")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Audio input
    audio = st.file_uploader("Upload a voice note (mp3/wav)", type=["mp3", "wav", "m4a"])
    if audio:
        user_text = transcribe_audio(audio)
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        maybe_update_draft(user_text)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Bharat Bot is thinking…")
            reply = llm_chat(user_text)
            placeholder.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Text input
    if prompt := st.chat_input("Type here…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        maybe_update_draft(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Bharat Bot is thinking…")
            reply = llm_chat(prompt)
            placeholder.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Sidebar
    with st.sidebar:
        st.header("Your Draft")
        st.json(st.session_state.draft, expanded=False)

        d = st.session_state.draft
        if draft_complete(d):
            if d["stage"] == "idea":
                st.success("All set! Let's schedule a call to dive deeper.")
                if st.button("Confirm – Schedule Call"):
                    st.info("We'll reach out soon. ✨")
            else:
                st.success("Ready to generate your one‑pager pitch deck!")
                if st.button("Generate Pitch Deck"):
                    md = generate_pitch_markdown(d)
                    pdf_path = markdown_to_pdf(md)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, "pitch_deck.pdf", "application/pdf")
        else:
            missing_fields = IDEA_SEQUENCE if d["stage"] == "idea" else MVP_SEQUENCE
            missing = [f for f in missing_fields if d.get(f) is None]
            st.info("Missing: " + ", ".join(missing))

    # Footer
    created = dt.datetime.fromisoformat(st.session_state.draft["created_at"])
    expires = created + dt.timedelta(days=DRAFT_EXPIRY_DAYS)
    st.caption(f"Draft stored until {expires.date()} (UTC)")


if __name__ == "__main__":
    main()
