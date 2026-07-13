import os
import json
import sys
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma

sys.path.append(os.path.dirname(__file__))
from core.vector_database import get_embeddings
from config.settings import VECTORSTORE_DIRECTORY

CURATED_DIR = os.path.join(os.path.dirname(__file__), "data", "curated")
PROVIDER = "groq"
BATCH_SIZE = 200


def load_clinical_trials(folder):
    docs = []
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    print(f"Processing {len(files)} clinical trial files...")
    for i, fname in enumerate(files):
        try:
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            protocol = data.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            desc = protocol.get("descriptionModule", {})
            conditions = protocol.get("conditionsModule", {})

            text = (
                f"Clinical Trial: {ident.get('briefTitle', 'N/A')}\n"
                f"NCT ID: {ident.get('nctId', 'N/A')}\n"
                f"Status: {status.get('overallStatus', 'N/A')}\n"
                f"Phase: {', '.join(design.get('phases', []) or ['N/A'])}\n"
                f"Conditions: {', '.join(conditions.get('conditions', []) or ['N/A'])}\n"
                f"Summary: {desc.get('briefSummary', 'N/A')}"
            )
            docs.append(Document(page_content=text, metadata={"source": fname, "type": "clinical_trial"}))
        except Exception:
            pass
        if i % 5000 == 0:
            print(f"  {i}/{len(files)} trials parsed")
    return docs


def load_faers(folder):
    docs = []
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    for fname in files:
        print(f"Processing FAERS file: {fname}")
        try:
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results", [])[:3000]  # sample per file
            for r in results:
                try:
                    patient = r.get("patient", {})
                    drugs = patient.get("drug", [])
                    reactions = patient.get("reaction", [])
                    drug_names = ", ".join([d.get("medicinalproduct", "N/A") for d in drugs][:5])
                    reaction_names = ", ".join([rx.get("reactionmeddrapt", "N/A") for rx in reactions][:5])
                    serious = "Yes" if r.get("serious") == "1" else "No"
                    text = (
                        f"Adverse Event Report\n"
                        f"Drug(s): {drug_names}\n"
                        f"Reaction(s): {reaction_names}\n"
                        f"Serious: {serious}\n"
                        f"Patient age: {patient.get('patientonsetage', 'N/A')}, sex: {patient.get('patientsex', 'N/A')}"
                    )
                    docs.append(Document(page_content=text, metadata={"source": fname, "type": "adverse_event"}))
                except Exception:
                    continue
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")
    return docs


def load_pubmed(filepath):
    docs = []
    if not os.path.exists(filepath):
        return docs
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    abstracts = content.split("\n\n\n")
    for a in abstracts:
        if len(a.strip()) > 50:
            docs.append(Document(page_content=a.strip(), metadata={"source": "pubmed", "type": "literature"}))
    print(f"Loaded {len(docs)} PubMed abstract chunks")
    return docs


def main():
    all_docs = []
    all_docs.extend(load_clinical_trials(os.path.join(CURATED_DIR, "clinical_trials")))
    all_docs.extend(load_faers(os.path.join(CURATED_DIR, "faers")))
    all_docs.extend(load_pubmed(os.path.join(CURATED_DIR, "pubmed_abstracts.txt")))

    print(f"\nTotal documents to embed: {len(all_docs)}")

    splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    embedding = get_embeddings(PROVIDER)
    persist_path = VECTORSTORE_DIRECTORY[PROVIDER]
    os.makedirs(persist_path, exist_ok=True)

    vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_path)

    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"  Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("\nIngestion complete. Vectorstore saved at:", persist_path)


if __name__ == "__main__":
    main()