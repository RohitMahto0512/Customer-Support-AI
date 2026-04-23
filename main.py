from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import ai_engine

# 1. Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./production_chat.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatRecord(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_message = Column(String)
    bot_response = Column(String)
    detected_intent = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)

# 2. FastAPI Setup
app = FastAPI(title="Customer Support AI API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 3. Pydantic Models for Data Validation
class ChatRequest(BaseModel):
    message: str

# 4. Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
def chat_endpoint(chat_request: ChatRequest):
    db = SessionLocal()
    try:
        # Get AI Prediction
        ai_result = ai_engine.get_ai_response(chat_request.message)

        # Save to Database
        new_log = ChatRecord(
            user_message=chat_request.message,
            bot_response=ai_result["response"],
            detected_intent=ai_result["intent"],
            sentiment=ai_result["sentiment"],
            confidence=ai_result["confidence"]
        )
        db.add(new_log)
        db.commit()
        return ai_result
    finally:
        db.close() # This will ALWAYS run, even if an error happens above
    
    return ai_result
