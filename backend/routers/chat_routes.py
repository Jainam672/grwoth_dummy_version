"""
Chat proxy route — forwards messages to Anthropic API server-side
so no API key is exposed in the browser.
"""
import os
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import urllib.request

from auth import get_current_user
import models

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    idea_context: Optional[str] = None


@router.post("/message")
def chat_message(
    payload: ChatRequest,
    current_user: models.User = Depends(get_current_user),
):
    """Proxy chat messages to AI backend (Anthropic or dummy)."""
    use_dummy = os.getenv("USE_DUMMY_AI", "true").lower() == "true"

    if use_dummy:
        last_msg = payload.messages[-1].content if payload.messages else ""
        response_text = _dummy_chat_response(last_msg, payload.idea_context)
        return {"response": response_text}

    # Real Anthropic API call
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Fallback to intelligent dummy if no key configured
        last_msg = payload.messages[-1].content if payload.messages else ""
        return {"response": _dummy_chat_response(last_msg, payload.idea_context)}

    system = (
        "You are GrowthPilot, an expert AI business mentor for Indian entrepreneurs. "
        "Give practical, actionable advice with specific numbers, timelines, and costs in INR. "
        "Use clear sections with bold headers and bullet points. "
        "Focus on Indian market conditions, regulations (GST, MSME, Udyam), and opportunities."
    )
    if payload.idea_context:
        system += f"\n\nContext: The user has submitted this business idea: {payload.idea_context}"

    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system,
        "messages": [{"role": m.role, "content": m.content} for m in payload.messages[-10:]]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            text = "".join(b.get("text","") for b in data.get("content",[]))
            return {"response": text}
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        last_msg = payload.messages[-1].content if payload.messages else ""
        return {"response": _dummy_chat_response(last_msg, payload.idea_context)}


def _dummy_chat_response(user_msg: str, context: str = None) -> str:
    lower = user_msg.lower()

    if context:
        return (
            f"**Based on your business idea** 📊\n\n"
            f"Great question! Here's my analysis:\n\n"
            f"**Key Insight:** Your idea has strong fundamentals. Here are my recommendations:\n\n"
            f"• **Market Validation:** Start by speaking with 10-15 potential customers in your target area\n"
            f"• **Registration:** Register under Udyam (MSME) for govt subsidies — completely free online\n"
            f"• **First 30 Days:** Focus on a minimal viable product (MVP) and get your first paying customer\n"
            f"• **Marketing:** Instagram + WhatsApp Business are the most cost-effective channels for Indian SMEs\n\n"
            f"**What specific aspect would you like to explore further?** I can dive deeper into costs, marketing, operations, or legal requirements."
        )

    if any(w in lower for w in ["chai","café","cafe","tea","restaurant","food"]):
        return (
            "**Chai Café / Food Business Analysis 🍵**\n\n"
            "**Market Overview:** Food & beverage is one of India's fastest-growing sectors. Chai cafés specifically have seen 34% growth in Tier-2 cities.\n\n"
            "**Setup Costs (₹2L budget):**\n"
            "• Equipment & appliances: ₹55,000\n"
            "• Interior & furniture: ₹40,000\n"
            "• First month ingredients: ₹20,000\n"
            "• Marketing launch: ₹15,000\n"
            "• Working capital reserve: ₹70,000\n\n"
            "**90-Day Roadmap:**\n"
            "📍 Month 1: FSSAI license, setup, soft launch to friends & family\n"
            "📍 Month 2: Instagram push, loyalty cards, 5-star Google reviews\n"
            "📍 Month 3: Swiggy/Zomato listing, expand menu, hire part-time staff\n\n"
            "**Break-even:** Typically 3-5 months at 60-70% capacity.\n\nWhat city are you targeting? I can give you more specific advice!"
        )

    if any(w in lower for w in ["tech","app","software","website","edtech","platform"]):
        return (
            "**Tech / Digital Business Strategy 💻**\n\n"
            "Tech businesses have excellent unit economics in India. Here's how to start:\n\n"
            "**Phase 1 — Validate (Week 1-4):**\n"
            "• Build a landing page (free on Carrd or Webflow)\n"
            "• Run ₹5,000 worth of Meta ads to test demand\n"
            "• Get 50 email sign-ups before building anything\n\n"
            "**Phase 2 — Build MVP (Month 1-3):**\n"
            "• Use no-code tools (Bubble, FlutterFlow) to reduce dev cost by 70%\n"
            "• Target: 10 paying beta users at ₹499-999/month\n\n"
            "**Phase 3 — Scale (Month 4+):**\n"
            "• Apply to Startup India for seed fund (up to ₹50L)\n"
            "• Content marketing on LinkedIn for B2B or Instagram for B2C\n\n"
            "**Revenue potential:** ₹50K-5L/month depending on niche. What's your specific idea?"
        )

    if any(w in lower for w in ["cost","budget","money","invest","capital","fund"]):
        return (
            "**Budget & Funding Guide 💰**\n\n"
            "Here are the key funding options for Indian entrepreneurs:\n\n"
            "**Bootstrap (₹50K - ₹5L):**\n"
            "• Best for: service businesses, freelancing, small retail\n"
            "• Keep overhead minimal, focus on cash-positive from day 1\n\n"
            "**Government Schemes:**\n"
            "• **Startup India Seed Fund:** Up to ₹50L for registered startups\n"
            "• **SIDBI CGTMSE:** Collateral-free loans up to ₹10L\n"
            "• **PM Mudra Yojana:** ₹10K to ₹10L for micro businesses\n"
            "• **Udyam Registration:** Free, unlocks subsidies & priority banking\n\n"
            "**Angel Funding (₹10L - ₹1Cr):**\n"
            "• Platforms: LetsVenture, AngelList India, 100X.VC\n"
            "• Need: traction, team, and a scalable model\n\n"
            "What's your business model? I can give a more specific breakdown."
        )

    if any(w in lower for w in ["market","customer","sell","sales","promote","advertise"]):
        return (
            "**Marketing Strategy for Indian SMEs 📣**\n\n"
            "Here's what actually works without burning your budget:\n\n"
            "**Digital (Low Cost, High ROI):**\n"
            "• WhatsApp Business: Build a broadcast list of 200+ contacts from day 1\n"
            "• Instagram Reels: 3 posts/week, behind-the-scenes + customer stories\n"
            "• Google My Business: Free listing, drives 40% of local walk-ins\n\n"
            "**Community Marketing:**\n"
            "• Join local Facebook groups and LinkedIn communities\n"
            "• Partner with complementary businesses for cross-promotion\n"
            "• Offer referral rewards (₹50-100 discount per referral)\n\n"
            "**Paid (Start small):**\n"
            "• Meta ads: Start with ₹200/day, test 3 creatives\n"
            "• Google Search ads: Only if people are searching for your service\n\n"
            "**Target CAC:** Should be less than 3x monthly revenue per customer."
        )

    return (
        "**AI Business Mentor Response 🤖**\n\n"
        "I'm here to help you build a successful business! Here's what I can assist with:\n\n"
        "• **Feasibility Analysis** — Is your idea viable in your market?\n"
        "• **Cost Estimation** — How much will you really need to invest?\n"
        "• **Growth Roadmap** — Step-by-step plan for your first 90 days\n"
        "• **Marketing Strategy** — How to get your first 100 customers\n"
        "• **Funding Options** — Government schemes, angels, bootstrapping\n\n"
        "**To get started:** Tell me about your business idea, your location, and your budget. "
        "The more specific you are, the more detailed my analysis will be!\n\n"
        "*Example: 'I want to open a cloud kitchen in Surat with ₹3L budget targeting office workers'*"
    )
