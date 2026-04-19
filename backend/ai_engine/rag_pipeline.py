"""
RAG Pipeline
-------------
1. Embed the user's business idea + question
2. Retrieve top-k relevant chunks from ChromaDB
3. Build a structured prompt
4. Pass to Phi-3-mini (or dummy LLM) and return parsed JSON
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

USE_DUMMY_AI = os.getenv("USE_DUMMY_AI", "true").lower() == "true"


def build_prompt(
    idea_title: str,
    idea_description: str,
    budget: str,
    location: str,
    category: str,
    experience_level: str,
    context_chunks: list,
) -> str:
    context_text = "\n\n".join(context_chunks) if context_chunks else "No additional context available."

    return f"""<|system|>
You are GrowthPilot, an expert AI business mentor helping entrepreneurs evaluate and plan their business ideas.
Always respond with a valid JSON object containing these exact keys:
feasibility, cost_breakdown, roadmap (list), marketing (list), risks (list),
competitors (list), funding (list), idea_score (0-100 integer), stage (string).
<|end|>

<|user|>
BUSINESS KNOWLEDGE CONTEXT:
{context_text}

BUSINESS IDEA:
Title: {idea_title}
Description: {idea_description}
Budget: {budget or "Not specified"}
Location: {location or "Not specified"}
Category: {category or "General"}
Founder Experience: {experience_level or "Beginner"}

Based on all the above, provide a comprehensive business analysis as a JSON object.
<|end|>

<|assistant|>
"""


def analyze_idea(
    idea_title: str,
    idea_description: str,
    budget: str = None,
    location: str = None,
    category: str = None,
    experience_level: str = "beginner",
) -> dict:
    """
    Full RAG pipeline:
    1. Embed idea
    2. Retrieve from ChromaDB
    3. Generate with LLM
    4. Parse and return structured dict
    """

    context_chunks = []

    if not USE_DUMMY_AI:
        # Step 1 — Embed query
        from ai_engine.embeddings import embed_text
        from ai_engine.vector_store import query_similar
        query = f"{idea_title}. {idea_description}"
        query_embedding = embed_text(query)

        # Step 2 — Retrieve
        context_chunks = query_similar(query_embedding, n_results=5)
        logger.info(f"Retrieved {len(context_chunks)} context chunks from ChromaDB.")

    # Step 3 — Build prompt and generate
    prompt = build_prompt(
        idea_title, idea_description,
        budget, location, category, experience_level,
        context_chunks
    )

    from ai_engine.llm import generate_response
    raw_output = generate_response(prompt)

    # Step 4 — Parse JSON from model output
    return _parse_output(raw_output)


def _parse_output(raw: str) -> dict:
    """Extract and parse JSON from model output."""
    try:
        # Try direct parse
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON block from text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Fallback if parsing fails
    logger.error("Failed to parse JSON from LLM output. Using fallback.")
    return {
        "feasibility": "Analysis complete. Please review the details.",
        "cost_breakdown": "Cost breakdown not available. Please refine your inputs.",
        "roadmap": ["Define MVP", "Build & test", "Launch", "Scale"],
        "marketing": ["Social media", "Word of mouth", "Content marketing"],
        "risks": ["Market risk", "Financial risk", "Execution risk"],
        "competitors": ["Research competitors in your niche"],
        "funding": ["Bootstrapping", "Angel investors", "Government grants"],
        "idea_score": 60,
        "stage": "Idea",
    }
