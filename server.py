import os
import json
from datetime import datetime
from bson import ObjectId
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient

# --- 1. CONFIGURATION & DATABASE CONNECTION ---
# Attempt to find the .env file dynamically
possible_paths = [
    "backend/sourceCode/.env",
    "sourceCode/.env",
    ".env"
]
loaded = False
for path in possible_paths:
    if os.path.exists(path):
        load_dotenv(path)
        print(f"✅ Loaded environment from: {path}")
        loaded = True
        break

if not loaded:
    print("⚠️ WARNING: Could not find .env file. API keys might be missing.")

api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_CONNECTION_STRING")

if not api_key:
    print("❌ ERROR: OPENAI_API_KEY is missing.")
if not mongo_uri:
    print("❌ ERROR: MONGO_URI is missing.")

client = OpenAI(api_key=api_key)

# Connect to MongoDB with error handling
try:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["medical_triage_db"]
    conversations_collection = db["Conversations"]
    mongo_client.admin.command('ping')
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATA MODELS ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

# --- 3. HELPER FUNCTIONS ---
def format_conversation_for_ai(messages):
    """Reconstructs the conversation history for the AI prompt."""
    history_str = ""
    for msg in messages:
        # Safety: Ensure we handle dicts correctly
        if isinstance(msg, dict):
            text = msg.get("text", "")
            sender = msg.get("sender", "")
            
            if sender == "user":
                if history_str == "":
                    history_str = f"Patient initial complaint: {text}"
                else:
                    history_str += f" | Answer: {text}"
            elif sender == "bot":
                if "Analysis Complete" not in text:
                     history_str += f" | Question: {text}"
    return history_str

def format_final_report(ai_data):
    """Generates the text report safely, even if AI format is slightly off."""
    report = "📋 **FINAL MEDICAL ANALYSIS**\n\n"
    
    if ai_data.get("candidates"):
        report += "Top Suspected Conditions:\n"
        candidates = ai_data["candidates"]
        
        # Sort if possible (only if it's a list of dicts with probability)
        try:
            candidates = sorted(candidates, key=lambda x: x.get('probability', 0), reverse=True)
        except:
            pass # If sorting fails, just use the list as is
            
        for c in candidates:
            # CRITICAL FIX: Check if 'c' is actually a dictionary before accessing keys
            if isinstance(c, dict) and 'condition' in c:
                prob = c.get('probability', 0)
                pct = round(prob * 100)
                report += f"• {c['condition']} ({pct}%)\n"
            elif isinstance(c, str):
                # Fallback if AI returns a simple list of strings
                report += f"• {c}\n"
    
    report += f"\n💡 **ADVICE:**\n{ai_data.get('top_recommendation', 'Please consult a doctor.')}"
    return report

# --- 4. ENDPOINTS ---

@app.get("/conversations")
def get_conversations():
    try:
        chats = conversations_collection.find().sort("created_at", -1)
        results = []
        for chat in chats:
            results.append({
                "id": str(chat["_id"]),
                "title": chat.get("title", "New Consultation"),
                "date": chat.get("created_at")
            })
        return results
    except Exception as e:
        print(f"Error fetching chats: {e}")
        return []

@app.get("/conversations/{conversation_id}")
def get_conversation_details(conversation_id: str):
    if not ObjectId.is_valid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid ID")
    
    chat = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {"messages": chat.get("messages", [])}

@app.post("/chat_step")
def chat_step(req: ChatRequest):
    try:
        # A. Retrieve or Create Chat
        if req.conversation_id:
            chat = conversations_collection.find_one({"_id": ObjectId(req.conversation_id)})
            if not chat: raise HTTPException(status_code=404, detail="Chat not found")
            conversation_id = ObjectId(req.conversation_id)
        else:
            new_chat = {
                "title": (req.message[:30] + "...") if len(req.message) > 30 else req.message,
                "created_at": datetime.now(),
                "messages": []
            }
            result = conversations_collection.insert_one(new_chat)
            conversation_id = result.inserted_id

        # B. Save User Message
        user_msg = {"sender": "user", "text": req.message, "timestamp": datetime.now().isoformat()}
        conversations_collection.update_one({"_id": conversation_id}, {"$push": {"messages": user_msg}})

        # C. AI Logic
        updated_chat = conversations_collection.find_one({"_id": conversation_id})
        messages_list = updated_chat["messages"]
        
        bot_question_count = sum(1 for m in messages_list if m["sender"] == "bot")
        history_prompt = format_conversation_for_ai(messages_list)
        
        # UPDATED PROMPT: Explicitly tells AI the structure of 'candidates'
        base_system = (
            "You are a medical triage system. "
            "Output a JSON object with keys: "
            "'candidates' (a list of objects, each having 'condition' string and 'probability' float), "
            "'next_question', 'top_recommendation'."
        )

        if bot_question_count >= 5:
            system_prompt = base_system + " You have collected enough info. Set 'next_question' to 'DIAGNOSIS_COMPLETE' and provide your final analysis."
        else:
            system_prompt = base_system + " If you have enough info, set 'next_question' to 'DIAGNOSIS_COMPLETE'. Otherwise, ask a relevant follow-up question."

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Patient History: {history_prompt}"}
            ],
            temperature=0.3
        )
        
        gpt_json_str = response.choices[0].message.content
        ai_data = json.loads(gpt_json_str)

        # D. Save Bot Response
        if ai_data.get("next_question") == "DIAGNOSIS_COMPLETE":
            bot_text = format_final_report(ai_data)
        else:
            bot_text = ai_data.get("next_question")

        bot_msg = {"sender": "bot", "text": bot_text, "timestamp": datetime.now().isoformat()}
        conversations_collection.update_one({"_id": conversation_id}, {"$push": {"messages": bot_msg}})

        return {"gpt_json": gpt_json_str, "conversation_id": str(conversation_id)}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}