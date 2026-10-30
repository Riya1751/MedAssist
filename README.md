# MedAssist – Phase 1 (Backend)

See the terminal commands in the chat reply. Quick start:

    brew install tesseract mysql python@3.12
    brew services start mysql
    mysql -u root < schema.sql
    python3.12 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then edit AI_API_KEY / AI_MODEL_NAME / AI_BASE_URL
    uvicorn app.main:app --reload --port 8000

Open https://medassist-2qmc.onrender.com/ for the interactive API tester.
