import re
import requests
from core.ml_tools import predict_severity, predict_trial_phase

SEVERITY_KEYWORDS = ["risk", "severity", "serious", "how dangerous", "how severe"]
PHASE_KEYWORDS = ["what phase", "predict phase", "which phase", "classify phase"]


def needs_severity_prediction(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SEVERITY_KEYWORDS)


def needs_phase_prediction(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in PHASE_KEYWORDS)


def extract_severity_features(query):
    drug_match = re.search(r'(\d+)\s*drugs?', query.lower())
    reaction_match = re.search(r'(\d+)\s*reactions?', query.lower())
    age_match = re.search(r'age\s*(\d+)', query.lower())
    num_drugs = int(drug_match.group(1)) if drug_match else 2
    num_reactions = int(reaction_match.group(1)) if reaction_match else 2
    age = int(age_match.group(1)) if age_match else 50
    return num_drugs, num_reactions, age


def fetch_live_pubmed(query, max_results=3):
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        resp = requests.get(search_url, params=params, timeout=10)
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"texts": [], "sources": []}

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        resp = requests.get(summary_url, params=params, timeout=10)
        summaries = resp.json().get("result", {})

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
        resp = requests.get(fetch_url, params=params, timeout=10)
        text = resp.text.strip()

        texts = [f"[Live PubMed result]\n{text[:1500]}"] if text else []
        sources = [
            {
                "title": summaries.get(pmid, {}).get("title") or f"PubMed article {pmid}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            for pmid in ids
        ]
        return {"texts": texts, "sources": sources}
    except Exception:
        return {"texts": [], "sources": []}


def fetch_live_clinicaltrials(query, max_results=3):
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {"query.term": query, "pageSize": max_results, "format": "json"}
        resp = requests.get(url, params=params, timeout=10)
        studies = resp.json().get("studies", [])
        texts = []
        sources = []
        for s in studies:
            protocol = s.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            title = ident.get("briefTitle", "N/A")
            nct_id = ident.get("nctId", "")
            texts.append(
                f"[Live ClinicalTrials.gov result]\n"
                f"Title: {title}\n"
                f"NCT ID: {nct_id}\n"
                f"Status: {status.get('overallStatus', 'N/A')}"
            )
            if nct_id:
                sources.append({"title": title, "url": f"https://clinicaltrials.gov/study/{nct_id}"})
        return {"texts": texts, "sources": sources}
    except Exception:
        return {"texts": [], "sources": []}


def run_severity_tool(num_drugs=2, num_reactions=2, age=50, sex=1):
    result = predict_severity(num_drugs, num_reactions, age, sex)
    if result is None:
        return None
    return (
        f"[ML Model Prediction - Decision Tree]\n"
        f"Input: {num_drugs} drug(s), {num_reactions} reaction(s), age {age}\n"
        f"Predicted severity: {result['prediction']} (confidence: {result['confidence']}%)"
    )


def run_phase_tool(text):
    result = predict_trial_phase(text)
    if result is None:
        return None
    return f"[ML Model Prediction - Neural Network]\nPredicted trial phase: {result['prediction']} (confidence: {result['confidence']}%)"