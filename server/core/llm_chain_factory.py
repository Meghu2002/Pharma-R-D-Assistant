from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from config.settings import GROQ_API_KEY, GOOGLE_API_KEY
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from utils.logger import logger
from core.agent_tools import (
        fetch_live_pubmed, fetch_live_clinicaltrials,
        needs_severity_prediction, needs_phase_prediction, run_severity_tool, run_phase_tool,
        extract_severity_features
)


def get_prompt():
    logger.debug("Creating chat prompt template.")
    return ChatPromptTemplate.from_messages([
        ("system", (
            "You are a pharma R&D research assistant. The context below may include retrieved documents, "
            "live internet search results, and machine learning model predictions (labeled accordingly). "
            "Use all of it together to give the most complete, well-reasoned answer possible. "
            "Only say 'I don't know' if the context truly contains nothing relevant to the question."
        )),
        ("human", "Context:\n{context}\n\n\nQuestion:\n{input}")
    ])


@lru_cache(maxsize=None)
def get_llm(model_provider: str, model: str):
    logger.debug(f"Initializing LLM for {model_provider} - {model}")
    if model_provider == "groq":
        return ChatGroq(model=model, api_key=GROQ_API_KEY)
    elif model_provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, api_key=GOOGLE_API_KEY)
    else:
        logger.error(f"Unsupported LLM Provider: {model_provider}")
        raise ValueError(f"Unsupported LLM Provider: {model_provider}")


def build_llm_chain(model_provider: str, model: str, vectorstore):
    logger.debug(f"Building LLM chain for provider: {model_provider}, model: {model}")
    prompt = get_prompt()
    llm = get_llm(model_provider, model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt=prompt)
    )


def run_agent_query(model_provider, model, vectorstore, message):
    local_docs = vectorstore.similarity_search(message, k=4)
    local_texts = [f"[Local knowledge base result]\n{d.page_content}" for d in local_docs]

    with ThreadPoolExecutor(max_workers=2) as executor:
        pubmed_future = executor.submit(fetch_live_pubmed, message)
        trials_future = executor.submit(fetch_live_clinicaltrials, message)
        pubmed_result = pubmed_future.result()
        trials_result = trials_future.result()
    live_texts = pubmed_result["texts"] + trials_result["texts"]
    live_sources = pubmed_result["sources"] + trials_result["sources"]

    ml_texts = []
    used_severity_model = needs_severity_prediction(message)
    used_phase_model = needs_phase_prediction(message)
    if used_severity_model:
        num_drugs, num_reactions, age = extract_severity_features(message)
        result = run_severity_tool(num_drugs=num_drugs, num_reactions=num_reactions, age=age)
        if result:
            ml_texts.append(result)
    if used_phase_model:
        result = run_phase_tool(message)
        if result:
            ml_texts.append(result)

    context = "\n\n".join(local_texts + live_texts + ml_texts)
    llm = get_llm(model_provider, model)
    prompt = get_prompt()
    formatted = prompt.format_messages(context=context, input=message)
    result = llm.invoke(formatted)
    answer = result.content if hasattr(result, "content") else str(result)

    trace = {
        "used_live_fetch": True,
        "used_severity_model": used_severity_model,
        "used_phase_model": used_phase_model,
        "local_chunks_found": len(local_docs),
        "live_chunks_found": len(live_texts),
        "live_sources": live_sources,
        "sources": [d.metadata for d in local_docs]
    }
    return answer, trace