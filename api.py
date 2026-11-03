
import base64
import io
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
import pypdf
from pydantic import BaseModel, Field

from src.predictor import predict
from auth import (
    get_current_user,
    get_optional_current_user,
    register_user,
    login_user,
    create_guest_session,
    logout_session,
    bearer,
)
import database
from database import (
    save_prediction,
    get_user_history,
    clear_user_history,
    create_session,
    save_chat_message,
    get_chat_history,
    clear_chat_history,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Load existing chatbot and backend settings.
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Cardio AI API",
    version="1.0.0",
    description=(
        "Cardio AI educational heart disease prediction "
        "and AI chat service."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Prediction request model
# --------------------------------------------------

class PatientData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bp: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_angina: int
    oldpeak: float
    st_segment: int
    major_vessels: int
    thal: int


# --------------------------------------------------
# Chat request models
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)


# --------------------------------------------------
# Python Authentication Models & Endpoints
# --------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = "User"


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/register")
def auth_register(payload: RegisterRequest):
    try:
        user = register_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
        )
        token = database.create_session(user["id"])
        return {
            "message": "Account created successfully!",
            "access_token": token,
            "user": user,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Registration error: {exc}")
        raise HTTPException(status_code=500, detail="Registration failed.")


@app.post("/auth/login")
def auth_login(payload: LoginRequest):
    try:
        return login_user(email=payload.email, password=payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        print(f"Login error: {exc}")
        raise HTTPException(status_code=500, detail="Login failed.")


@app.post("/auth/free-trial")
def auth_free_trial():
    try:
        return create_guest_session()
    except Exception as exc:
        print(f"Free trial error: {exc}")
        raise HTTPException(status_code=500, detail="Could not create free trial session.")


@app.get("/auth/me")
def auth_me(current_user=Depends(get_current_user)):
    return {"user": current_user.to_dict()}


@app.post("/auth/logout")
def auth_logout(credentials=Depends(bearer)):
    if credentials and credentials.credentials:
        logout_session(credentials.credentials)
    return {"message": "Logged out successfully"}


# --------------------------------------------------
# General API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Cardio AI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction endpoint (requires authentication)
# --------------------------------------------------

class HistoryRecord(BaseModel):
    patient_data: dict
    prediction_data: dict


@app.post("/predict")
def predict_patient(
    data: PatientData,
    current_user=Depends(get_optional_current_user),
):
    patient_data = {
        "age": data.age,
        "sex": data.sex,
        "chest pain type": data.chest_pain_type,
        "resting blood pressure": data.resting_bp,
        "serum cholestoral": data.cholesterol,
        "fasting blood sugar": data.fasting_blood_sugar,
        "resting electrocardiographic results": data.resting_ecg,
        "max heart rate": data.max_heart_rate,
        "exercise induced angina": data.exercise_angina,
        "oldpeak": data.oldpeak,
        "ST segment": data.st_segment,
        "major vessels": data.major_vessels,
        "thal": data.thal,
    }

    try:
        result = predict(patient_data)
    except Exception as exc:
        print(f"Prediction failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    history_id = None
    if current_user and getattr(current_user, "id", None) and current_user.id != "guest":
        try:
            record = save_prediction(
                user_id=current_user.id,
                patient_data=patient_data,
                prediction=result,
            )
            if record and "id" in record:
                history_id = str(record["id"])
        except Exception as db_exc:
            print(f"Warning: Could not save prediction to database: {db_exc}")

    return {
        "prediction": result,
        "history_id": history_id,
        "disclaimer": (
            "Educational model output only. "
            "This is not a clinical diagnosis."
        ),
    }


# --------------------------------------------------
# Prediction history endpoints
# --------------------------------------------------

@app.post("/history")
def save_prediction_history(
    payload: HistoryRecord,
    current_user=Depends(get_optional_current_user),
):
    if current_user and getattr(current_user, "id", None) and current_user.id != "guest":
        try:
            record = save_prediction(
                user_id=current_user.id,
                patient_data=payload.patient_data,
                prediction=payload.prediction_data,
            )
            return {"status": "ok", "id": str(record["id"]) if record else None}
        except Exception as exc:
            print(f"Warning: History save failed: {exc}")
            return {"status": "warning", "detail": "Database unavailable"}
    return {"status": "skipped", "detail": "Guest or unauthenticated"}


@app.get("/history")
def prediction_history(
    current_user=Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    try:
        records = get_user_history(
            user_id=current_user.id,
            limit=limit,
            offset=offset,
        )
        return {"history": records}
    except Exception as exc:
        print(f"Warning: Failed to retrieve history: {exc}")
        return {"history": []}


@app.delete("/history")
def delete_prediction_history(
    current_user=Depends(get_current_user),
):
    try:
        clear_user_history(user_id=current_user.id)
        return {"status": "ok", "message": "History cleared"}
    except Exception as exc:
        print(f"Warning: Failed to clear history: {exc}")
        raise HTTPException(status_code=500, detail="Failed to clear history")


# --------------------------------------------------
# Cardio AI chatbot
# --------------------------------------------------

CARDIO_AI_INSTRUCTIONS = """
You are Cardio AI, a friendly educational assistant for a heart
disease prediction project.

You can explain:
- The purpose of the Cardio AI application.
- Decision Tree and Bayesian Network concepts.
- How the application's prediction interface works.
- General heart-health educational information.

Conversation:
- Use the supplied conversation history for follow-up questions.
- Respond clearly and helpfully.
- Do not claim access to local files, the dataset, trained models,
  or actual patient prediction results unless that information
  is explicitly supplied.
- If you do not know a project-specific fact, say so.

Safety:
- Do not diagnose a person or prescribe treatment.
- Do not present generated answers as clinically validated advice.
- For potentially urgent symptoms, such as severe chest pain or
  difficulty breathing, advise seeking emergency medical care.
"""


@app.post("/chat")
def chat(
    request: ChatRequest,
    current_user=Depends(get_optional_current_user),
):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing from the backend environment.",
        )

    if not model_name:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_MODEL is missing from the backend environment.",
        )

    messages = []

    for item in request.history[-12:]:
        content = item.content.strip()

        if content:
            messages.append({
                "role": item.role,
                "content": content,
            })

    messages.append({
        "role": "user",
        "content": request.message.strip(),
    })

    try:
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=model_name,
            instructions=CARDIO_AI_INSTRUCTIONS,
            input=messages,
            max_output_tokens=700,
        )

        answer = response.output_text

        if not answer or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="The AI provider returned an empty response.",
            )

        if current_user and getattr(current_user, "id", None) and current_user.id != "guest":
            try:
                save_chat_message(user_id=current_user.id, role="user", text=request.message.strip())
                save_chat_message(user_id=current_user.id, role="assistant", text=answer.strip())
            except Exception as exc:
                print(f"Warning: Failed to save chat history: {exc}")

        return {"answer": answer}

    except OpenAIError as exc:
        print(
            f"Cardio AI OpenAI error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI provider request failed. Check the backend "
                "terminal, API key, account access, and model name."
            ),
        ) from exc


@app.post("/chat/report")
async def chat_report(
    file: UploadFile = File(...),
    message: str = Form(""),
    history: str = Form("[]"),
    current_user=Depends(get_optional_current_user),
):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing from the backend environment.",
        )

    if not model_name:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_MODEL is missing from the backend environment.",
        )

    filename = file.filename or "uploaded_report"
    ext = Path(filename).suffix.lower()
    allowed_exts = {".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg"}

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}.",
        )

    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 10 MB limit.")

    try:
        parsed_history = json.loads(history) if history else []
        if not isinstance(parsed_history, list):
            parsed_history = []
    except Exception:
        parsed_history = []

    messages = []
    for item in parsed_history[-10:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})

    user_query = message.strip() or "Please analyse this report and summarise the key findings in simple terms."

    if ext in {".png", ".jpg", ".jpeg"}:
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        b64_img = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_img}"
        user_message_content = [
            {"type": "input_image", "image_url": data_url},
            {
                "type": "input_text",
                "text": f"Attached image report: {filename}\n\nUser request: {user_query}",
            },
        ]
    else:
        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    pt = page.extract_text() or ""
                    if pt.strip():
                        pages_text.append(f"--- Page {idx + 1} ---\n{pt.strip()}")
                extracted_text = "\n\n".join(pages_text).strip()
                if not extracted_text:
                    extracted_text = "[Notice: This PDF does not contain extractable machine-readable text. It may contain scanned image pages.]"
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {exc}")
        else:
            extracted_text = raw_bytes.decode("utf-8", errors="replace").strip()

        if len(extracted_text) > 15000:
            extracted_text = extracted_text[:15000] + "\n\n[... Remaining content truncated for length ...]"

        user_message_content = (
            f"User provided an attached report file: '{filename}'.\n\n"
            f"--- REPORT CONTENT START ---\n"
            f"{extracted_text}\n"
            f"--- REPORT CONTENT END ---\n\n"
            f"User Question:\n{user_query}"
        )

    messages.append({
        "role": "user",
        "content": user_message_content,
    })

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model_name,
            instructions=CARDIO_AI_INSTRUCTIONS,
            input=messages,
            max_output_tokens=850,
        )

        answer = response.output_text
        if not answer or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="The AI provider returned an empty response.",
            )

        if current_user and getattr(current_user, "id", None) and current_user.id != "guest":
            try:
                user_msg = f"📎 {filename}\n{user_query}" if filename else user_query
                save_chat_message(user_id=current_user.id, role="user", text=user_msg)
                save_chat_message(user_id=current_user.id, role="assistant", text=answer.strip())
            except Exception as exc:
                print(f"Warning: Failed to save report chat history: {exc}")

        return {"answer": answer}

    except OpenAIError as exc:
        print(f"Cardio AI OpenAI report error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"The AI provider request failed ({type(exc).__name__}). Check OpenAI settings.",
        ) from exc
    except Exception as exc:
        print(f"Cardio AI unexpected report error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------
# Chat history endpoints
# --------------------------------------------------

@app.get("/chat/history")
def get_user_chat_history(
    current_user=Depends(get_current_user),
    limit: int = Query(default=100, ge=1, le=200),
):
    try:
        messages = get_chat_history(user_id=current_user.id, limit=limit)
        return {"messages": messages}
    except Exception as exc:
        print(f"Warning: Failed to retrieve chat history: {exc}")
        return {"messages": []}


@app.delete("/chat/history")
def clear_user_chat_history(
    current_user=Depends(get_current_user),
):
    try:
        clear_chat_history(user_id=current_user.id)
        return {"status": "ok", "message": "Chat history cleared"}
    except Exception as exc:
        print(f"Warning: Failed to clear chat history: {exc}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")


