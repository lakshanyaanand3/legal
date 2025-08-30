import os, json, re, time, sys, concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from pypdf import PdfReader
from docx import Document

# ------------------ Setup ------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------ Prompt Builder ------------------
def build_prompt(text: str) -> str:
    return f"""
Identify legal jargon in the following text and explain each term in simple Indian legal context.

Return ONLY JSON with this schema:
{{
  "terms": [
    {{
      "term": "string",
      "definition": "2-3 line meaning",
      "category": "contract|procedure|criminal|civil|property|constitutional|evidence|misc"
    }}
  ]
}}

Text:
\"\"\"{text}\"\"\"
"""

# ------------------ LLM Call ------------------
def extract_jargon(text: str):
    prompt = build_prompt(text)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Indian law."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"terms": []}

# ------------------ Chunking ------------------
def chunk_text(text, chunk_size=3000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_chunks_and_merge(text, use_chars=True, chunk_size_chars=3000, overlap_chars=200, concurrency=3):
    chunks = chunk_text(text, chunk_size_chars, overlap_chars)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(extract_jargon, chunk) for chunk in chunks]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if "terms" in res:
                results.extend(res["terms"])

    # Merge duplicates by term name
    merged = {}
    for term in results:
        tname = term["term"].lower()
        if tname not in merged:
            merged[tname] = term
        else:
            # merge definitions if different
            if term["definition"] not in merged[tname]["definition"]:
                merged[tname]["definition"] += " | " + term["definition"]
    return list(merged.values())

# ------------------ Save & Print ------------------
def save_results(terms, source_file):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    json_path = outputs_dir / f"jargon_{Path(source_file).stem}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": source_file, "terms": terms}, f, ensure_ascii=False, indent=2)

    rprint(f"[green]Saved results to[/green] {json_path}")

    # Pretty print
    table = Table(title="Identified Legal Jargon")
    table.add_column("Term", style="bold")
    table.add_column("Definition")
    table.add_column("Category")
    for t in terms:
        table.add_row(t["term"], t["definition"], t["category"])
    rprint(table)

# ------------------ File Reader ------------------
def read_file(input_file: str) -> str:
    ext = Path(input_file).suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(input_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    elif ext == ".txt":
        return Path(input_file).read_text(encoding="utf-8")

    elif ext == ".docx":
        doc = Document(input_file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        rprint(f"[red]Unsupported file type: {ext}. Use .txt, .pdf or .docx[/red]")
        sys.exit(1)

# ------------------ Main ------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        rprint("[red]Usage: python main.py <path_to_file.txt|file.pdf|file.docx>[/red]")
        sys.exit(1)

    input_file = sys.argv[1]
    text = read_file(input_file)

    merged_results = process_chunks_and_merge(
        text,
        use_chars=True,
        chunk_size_chars=3000,
        overlap_chars=200,
        concurrency=3
    )

    save_results(merged_results, input_file)
