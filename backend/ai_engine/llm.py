"""
LLM Module — Phi-3-mini with optional LoRA adapter.

When USE_DUMMY_AI=true in .env, a structured dummy response is returned
so you can develop/test the platform without a GPU.
"""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

USE_DUMMY_AI = os.getenv("USE_DUMMY_AI", "true").lower() == "true"
MODEL_NAME   = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
LORA_PATH    = os.getenv("LORA_WEIGHTS_PATH", "")

_pipeline = None   # singleton model pipeline


def _load_model():
    """Load Phi-3-mini + optional LoRA weights (called once on first request)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info(f"Loading model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if LORA_PATH and os.path.exists(LORA_PATH):
        logger.info(f"Applying LoRA weights from: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)

    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    logger.info("Model loaded successfully.")
    return _pipeline


def generate_response(prompt: str) -> str:
    """Generate text from prompt. Uses dummy data if USE_DUMMY_AI=true."""
    if USE_DUMMY_AI:
        return _dummy_response()

    pipe = _load_model()
    outputs = pipe(prompt, return_full_text=False)
    return outputs[0]["generated_text"].strip()


def _dummy_response() -> str:
    """Structured JSON dummy response for development/testing."""
    return json.dumps({
        "feasibility": (
            "This business idea shows strong potential. The market demand is validated "
            "by current industry trends and consumer behavior. Execution feasibility is "
            "medium-high — the main challenge lies in initial customer acquisition, "
            "which can be addressed with targeted digital marketing."
        ),
        "cost_breakdown": (
            "Estimated startup costs: Registration & Legal ₹15,000 | Website & Tech ₹25,000 | "
            "Marketing (first 3 months) ₹30,000 | Operations & Inventory ₹50,000 | "
            "Miscellaneous ₹10,000. Total estimated: ₹1,30,000."
        ),
        "roadmap": [
            "Month 1: Legal registration, brand identity, website development",
            "Month 2: Soft launch, onboard first 10 customers, gather feedback",
            "Month 3: Refine product/service based on feedback, ramp up marketing",
            "Month 4-6: Scale operations, explore B2B partnerships",
            "Month 7-12: Target breakeven, launch referral program"
        ],
        "marketing": [
            "Leverage Instagram Reels and YouTube Shorts for organic reach",
            "Partner with local micro-influencers for trust-building",
            "Offer a freemium or trial tier to reduce entry barrier",
            "Build an email list from day one using lead magnets",
            "List on Google My Business for local discoverability"
        ],
        "risks": [
            "High competition from established players — mitigate with niche focus",
            "Cash flow issues in early months — maintain 3-month operating reserve",
            "Customer acquisition cost may exceed projections — A/B test ad creatives",
            "Regulatory changes — consult a local legal advisor quarterly"
        ],
        "competitors": [
            "Market Leader Corp — strong brand but poor customer service",
            "StartupX — tech-forward but expensive pricing",
            "LocalPro — limited to metro cities, your opportunity in Tier-2"
        ],
        "funding": [
            "Bootstrapping — ideal for MVP stage with ₹1-2L budget",
            "Startup India Seed Fund — up to ₹50L for registered startups",
            "Angel investors via platforms like LetsVenture or AngelList India",
            "SIDBI CGTMSE loan — collateral-free up to ₹10L for new businesses"
        ],
        "idea_score": 74,
        "stage": "MVP"
    })
