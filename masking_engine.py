import sys
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from gliner import GLiNER

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_loading = False

async def load_model():
    global model, model_loading
    if model is None and not model_loading:
        model_loading = True
        print("Laddar AI-modellen... Detta kan ta en stund på en ny maskin.", flush=True)
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(None, GLiNER.from_pretrained, "urchade/gliner_multi-v2.1")
            print("AI-modellen är laddad och redo!", flush=True)
        except Exception as e:
            print(f"Fel vid nerladdning av AI: {e}", flush=True)
        finally:
            model_loading = False

@app.on_event("startup")
async def startup_event():
    print("API server startad. Initierar AI...", flush=True)
    asyncio.create_task(load_model())

class AnalyzeRequest(BaseModel):
    text: str
    entities: List[str]

class Match(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str

def find_regex_matches(text: str) -> List[Match]:
    matches = []
    pnr_pattern = r'\b(?:\d{2})?\d{6}[-+]?\d{4}\b'
    for m in re.finditer(pnr_pattern, text):
        matches.append(Match(text=m.group(), label="Personnummer", start=m.start(), end=m.end(), source="regex"))
        
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for m in re.finditer(email_pattern, text):
        matches.append(Match(text=m.group(), label="E-post", start=m.start(), end=m.end(), source="regex"))
        
    phone_pattern = r'(?:(?:0|\+46|0046)\s?(?:[1-9]\d{1,2}\s?(?:[- ]?\d{2,3}){1,3}|\d{2,3}\s?(?:[- ]?\d{2,3}){1,3}))'
    for m in re.finditer(phone_pattern, text):
        matches.append(Match(text=m.group().strip(), label="Telefonnummer", start=m.start(), end=m.start() + len(m.group().strip()), source="regex"))

    return matches

@app.post("/analyze", response_model=List[Match])
def analyze_text(req: AnalyzeRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="AI-modellen laddas fortfarande ner, vänligen vänta...")

    regex_matches = find_regex_matches(req.text)
    labels = req.entities
    gliner_matches = []
    
    if labels:
        try:
            predictions = model.predict_entities(req.text, labels, threshold=0.4)
            for p in predictions:
                gliner_matches.append(Match(text=p["text"], label=p["label"], start=p["start"], end=p["end"], source="gliner"))
        except Exception as e:
            print(f"Error running model: {e}", flush=True)
            
    all_matches = regex_matches + gliner_matches
    all_matches.sort(key=lambda x: x.start)
    
    filtered = []
    last_end = -1
    for m in all_matches:
        if m.start >= last_end:
            filtered.append(m)
            last_end = m.end

    return filtered

if __name__ == "__main__":
    port = 8594
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Startar uvicorn på {port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
