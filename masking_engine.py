import sys
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from transformers import pipeline

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

def init_pipeline():
    return pipeline("ner", model="KBLab/bert-base-swedish-cased-ner", aggregation_strategy="simple")

async def load_model():
    global model, model_loading
    if model is None and not model_loading:
        model_loading = True
        print("Laddar KBLab Svensk-motor... Detta kan ta en stund på en ny maskin.", flush=True)
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(None, init_pipeline)
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
        matches.append(Match(text=m.group().strip(), label="Telefonnummer", start=m.start(), end=m.end() - (len(m.group()) - len(m.group().strip())), source="regex"))

    return matches

@app.post("/analyze", response_model=List[Match])
def analyze_text(req: AnalyzeRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="AI-modellen laddas fortfarande ner, vänligen vänta...")

    regex_matches = find_regex_matches(req.text)
    
    ui_to_ner = {
        "Person": "PER",
        "Organisation": "ORG",
        "Plats": "LOC",
        "Övrigt": "MISC"
    }
    ner_to_ui = {v: k for k, v in ui_to_ner.items()}
    
    requested_ner_tags = [ui_to_ner[e] for e in req.entities if e in ui_to_ner]
    
    kb_matches = []
    
    if requested_ner_tags:
        # Säkerhetsmekanism: Dela upp extremt långa texter i mindre bitar 
        # (BERT klarar max 512 tokens åt gången, ca 1500-2000 tecken beroende på ordlängd)
        max_chars = 1500
        chunks = []
        chunk_starts = []
        current_start = 0
        
        while current_start < len(req.text):
            end = current_start + max_chars
            if end >= len(req.text):
                end = len(req.text)
            else:
                # Försök bryta vid en punkt eller mellanslag så att vi inte klipper ord
                cut_idx = req.text.rfind('. ', current_start, end)
                if cut_idx == -1:
                    cut_idx = req.text.rfind(' ', current_start, end)
                
                if cut_idx != -1 and cut_idx > current_start:
                    end = cut_idx + 1 # Klipp precis efter punkten/mellanslaget
                    
            chunks.append(req.text[current_start:end])
            chunk_starts.append(current_start)
            current_start = end

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                predictions = model(chunk)
                for p in predictions:
                    entity_group = p.get("entity_group")
                    if entity_group in requested_ner_tags:
                        word = str(p.get("word", "")).replace("##", "")
                        # Returnerade indices är relativa till Chunken, addera chunkens start position
                        global_start = chunk_starts[i] + p["start"]
                        global_end = chunk_starts[i] + p["end"]
                        
                        kb_matches.append(Match(
                            text=word, 
                            label=ner_to_ui[entity_group], 
                            start=global_start, 
                            end=global_end, 
                            source="kblab"
                        ))
            except Exception as e:
                print(f"Fel vid körning av modell på chunk {i}: {e}", flush=True)
            
    all_matches = regex_matches + kb_matches
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
