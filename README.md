Reads input file (PDF/DOCX/TXT).

Splits into chunks of ~1500 words with 100-word overlap for context.

Sends each chunk to the LLM for jargon extraction.

Merges and removes duplicates.

Saves final JSON + prints a nice table.
cd D:\legal\legal_jargon_project\scripts
python main.py ..\data\sample_contract.txt
