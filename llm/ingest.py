import pdfplumber 
import os
import re
import json
from wordfreq import tokenize

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        full_text = []
        for page in pdf.pages:
            width = page.width
            
            # Détection heuristique : y a-t-il un "vide" au milieu de la page ?
            mid_zone = page.crop((width*0.45, 0, width*0.55, page.height))
            mid_words = mid_zone.extract_words()
            
            if len(mid_words) < 3:  # peu de mots au centre → double colonne
                left = page.crop((0, 0, width/2, page.height))
                right = page.crop((width/2, 0, width, page.height))
                left_words = left.extract_words(x_tolerance=3, y_tolerance=3)
                right_words = right.extract_words(x_tolerance=3, y_tolerance=3)
                page_text = " ".join([w["text"] for w in left_words])
                page_text += " " + " ".join([w["text"] for w in right_words])
            else:  # colonne unique
                words = page.extract_words(x_tolerance=3, y_tolerance=3)
                page_text = " ".join([w["text"] for w in words])
            
            full_text.append(page_text)
        return "\n".join(full_text)

def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def split_merged_words(text):
    """Seperates words that are merged together without spaces, often due to OCR errors."""
    words = text.split()
    result = []
    for word in words:
        # Si le mot est anormalement long et tout en minuscules → suspect
        if len(word) > 20 and word.islower():
            # Tokenize tentera de le découper intelligemment
            tokens = tokenize(word, 'en')
            result.extend(tokens)
        else:
            result.append(word)
    return " ".join(result)

def clean_text_proper(text):
    text = re.sub(r'-\n', '', text)         
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'Copyright.*?\.|Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x00-\x7FÀ-ÿ]+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    text = split_merged_words(text)  # ✅ ici
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":

    all_chunks = []

    for file in os.listdir("data/raw"):
        if file.endswith(".pdf"):
            path = os.path.join("data/raw", file)
            text = extract_text_from_pdf(path)
            text = clean_text_proper(text)  
            chunks = chunk_text(text)       
            all_chunks.extend(chunks)

    print(f"Nombre total de chunks : {len(all_chunks)}")
    print("Exemple :", all_chunks[0][:300])

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(all_chunks)} chunks saved.")