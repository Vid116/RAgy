# Asistent za Avtomobilska Zavarovanja

Sistem za inteligentno svetovanje o avtomobilskih zavarovanjih, ki temelji na RAG (Retrieval-Augmented Generation) arhitekturi. Sistem uporablja vektorsko bazo podatkov za shranjevanje in iskanje relevantnih informacij o zavarovanjih ter napredne jezikovne modele za generiranje odgovorov.

## 🚀 Funkcionalnosti

- Inteligentno svetovanje o avtomobilskih zavarovanjih
- Obdelava in indeksiranje dokumentov različnih formatov (PDF, DOCX, TXT, HTML, itd.)
- Vektorska baza podatkov za hitro in natančno iskanje informacij
- Spletni vmesnik za komunikacijo z asistentom
- Podpora za slovenski jezik

## 💻 Tehnične Zahteve

- Python 3.8 ali novejši
- OpenAI API ključ
- Ustrezne knjižnice (glej `requirements.txt`)

## 🛠️ Namestitev

1. Klonirajte repozitorij:
```bash
git clone [url-repozitorija]
```

2. Namestite potrebne knjižnice:
```bash
pip install -r requirements.txt
```

3. Ustvarite `.env` datoteko in dodajte potrebne spremenljivke okolja:
```env
OPENAI_API_KEY=your-api-key-here
```

## 📁 Struktura Projekta

- `DbUploader.py` - Razred za nalaganje dokumentov v vektorsko bazo
- `Talk_Vector.py` - Glavni razred za RAG funkcionalnost in komunikacijo z LLM
- `Prompts.py` - Predloge za generiranje poizvedb
- `rag-web-backend.py` - FastAPI spletni strežnik
- `index.html` - Osnovni spletni vmesnik
- `remove_from_db.py` - Orodje za odstranjevanje dokumentov iz baze

## 🚦 Uporaba

### Nalaganje Dokumentov

```python
from DbUploader import DocumentUploader

uploader = DocumentUploader()
db = uploader.upload_directory(
    directory="pot/do/dokumentov",
    db_directory="./vector_db_MD",
    collection_name="Car_stuff",
    recursive=True
)
```

### Zagon Asistenta

```python
from Talk_Vector import RAGConversationAgent

agent = RAGConversationAgent(
    db_path="./vector_db_MD",
    collection_name="Car_stuff",
    model_name="gpt-4"
)
agent.start_interactive_chat()
```

### Zagon Spletnega Vmesnika

```bash
python rag-web-backend.py
```

## 🔧 Konfiguracija

Sistem omogoča prilagajanje različnih parametrov:
- Velikost dokumentnih odsekov (`chunk_size`)
- Prekrivanje odsekov (`chunk_overlap`)
- Temperatura generiranja (`temperature`)
- Število dokumentov za iskanje (`k`)
- Dolžina zgodovine pogovora (`max_history_length`)



## 📝 Opombe

- Sistem je optimiziran za slovenski jezik in specifično domeno avtomobilskih zavarovanj
- Za optimalno delovanje je priporočljiva uporaba GPT-4 modela
- Vsi dokumenti v bazi morajo biti v slovenskem jeziku

## ⚖️ Licenca

BUREK
