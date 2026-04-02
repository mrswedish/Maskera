import sys
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from gliner import GLiNER

app = FastAPI()

# Ladda GLiNER-modellen (laddas ner automatiskt första gången)
model = GLiNER.from_pretrained("knowledgator/gliner-pii-small-v1.0")

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
    
    # Svenskt Personnummer (ååmmdd-xxxx, ååååmmdd-xxxx, eller utan bindestreck)
    pnr_pattern = r'\b(?:\d{2})?\d{6}[-+]?\d{4}\b'
    for m in re.finditer(pnr_pattern, text):
        matches.append(Match(
            text=m.group(),
            label="Personnummer",
            start=m.start(),
            end=m.end(),
            source="regex"
        ))
        
    # E-post
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for m in re.finditer(email_pattern, text):
        matches.append(Match(
            text=m.group(),
            label="E-post",
            start=m.start(),
            end=m.end(),
            source="regex"
        ))
        
    # Telefonnummer (Svenska)
    # Matchar +46, 0046, 0..., med eller utan mellanslag och bindestreck
    phone_pattern = r'(?:(?:0|\+46|0046)\s?(?:[1-9]\d{1,2}\s?(?:[- ]?\d{2,3}){1,3}|\d{2,3}\s?(?:[- ]?\d{2,3}){1,3}))'
    for m in re.finditer(phone_pattern, text):
        matches.append(Match(
            text=m.group().strip(),
            label="Telefonnummer",
            start=m.start(),
            # Räkna ut end manuellt eftersom strip kan ha tagit bort whitespaces i slutet
            end=m.start() + len(m.group().strip()), 
            source="regex"
        ))

    return matches

@app.post("/analyze", response_model=List[Match])
def analyze_text(req: AnalyzeRequest):
    # Regex-sökningar
    regex_matches = find_regex_matches(req.text)
    
    # GLiNER-sökning
    # the frontend passes wanted entities like: ["Person", "Organization", "Location"]
    labels = req.entities
    gliner_matches = []
    if labels:
        try:
            predictions = model.predict_entities(req.text, labels, threshold=0.4)
            for p in predictions:
                gliner_matches.append(Match(
                    text=p["text"],
                    label=p["label"],
                    start=p["start"],
                    end=p["end"],
                    source="gliner"
                ))
        except Exception as e:
            print(f"Error running model: {e}")
            
    # Slå ihop och filtrera överlappningar (enkelt sätt, prioritera regex)
    all_matches = regex_matches + gliner_matches
    
    # Sortera efter start-index och hantera överlappning
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
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
