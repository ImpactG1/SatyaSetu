"""
Groq LLM Service — Intelligent Reasoning for Misinformation Analysis
Uses Groq's ultra-fast inference for generating human-quality explanations,
source attribution, and deep reasoning about content veracity.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class GroqReasoningService:
    """
    Uses Groq LLM API for:
    1. Deep reasoning about why content is/isn't misinformation
    2. Source cross-referencing — "According to Reuters, Hindustan Times..."
    3. Generating actionable, professional explanations
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("Groq API key not configured — LLM reasoning disabled")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call_groq(self, messages: List[Dict], temperature: float = 0.3,
                   max_tokens: int = 1024, model: str = "llama-3.3-70b-versatile") -> Optional[str]:
        """Make a call to the Groq API."""
        if not self.api_key:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.Timeout:
            logger.error("Groq API request timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"Groq API request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected Groq API response format: {e}")
            return None

    def generate_deep_reasoning(
        self,
        title: str,
        text: str,
        signal_scores: Dict,
        key_indicators: List[Dict],
        fact_check_results: List[Dict],
        topic_info: Dict,
        misinformation_likelihood: float,
        risk_level: str,
    ) -> Optional[str]:
        """
        Generate a professional, detailed AI reasoning explanation.
        Returns human-quality paragraph explaining WHY this content is or isn't misinformation.
        """
        # Build context for the LLM
        indicators_text = ""
        for ind in key_indicators[:6]:
            indicators_text += f"  - [{ind.get('type', 'unknown')}] {ind.get('description', '')}\n"

        fc_text = "None found."
        if fact_check_results:
            fc_parts = []
            for fc in fact_check_results[:4]:
                publisher = fc.get('publisher', 'Unknown')
                rating = fc.get('rating', 'Unrated')
                claim = fc.get('claim_text', '')[:120]
                fc_parts.append(f"  - {publisher}: \"{claim}\" → Rating: {rating}")
            fc_text = "\n".join(fc_parts)

        topics_text = ", ".join(topic_info.get('labels', [])) or "General"

        pct = round(misinformation_likelihood * 100, 1)

        system_prompt = (
            "You are SatyaSetu AI — a misinformation detection expert. "
            "Generate a concise, professional analysis (3-5 sentences) explaining "
            "your assessment. Reference specific sources by name when available "
            "(e.g., 'According to Hindustan Times...', 'Reuters reports...', "
            "'IndiaToday Fact Check found...'). Be specific about what signals "
            "raised or lowered suspicion. Do NOT use markdown. Write in plain text."
        )

        user_prompt = f"""Analyze this content and explain your verdict in 3-5 clear sentences.

CONTENT TITLE: {title}
CONTENT TEXT: {text[:600]}

VERDICT: {pct}% misinformation likelihood ({risk_level.upper()} risk)

DETECTION SIGNALS:
  Claim Plausibility Score: {signal_scores.get('plausibility', 0):.2f}
  Linguistic Red Flags: {signal_scores.get('linguistic', 0):.2f}
  Source Quality Score: {signal_scores.get('source', 0):.2f}
  Fact-Check Cross-Ref: {signal_scores.get('fact_check', 0):.2f}
  Topic Sensitivity: {signal_scores.get('topic', 0):.2f}

KEY INDICATORS:
{indicators_text}

FACT-CHECK RESULTS:
{fc_text}

SENSITIVE TOPICS: {topics_text}

Write a clear, professional explanation that:
1. States the overall verdict and confidence
2. References specific sources/fact-checkers by name if available
3. Explains the most important red flags or credibility signals
4. Notes the topic sensitivity and potential societal impact
Keep it factual and authoritative. No speculation."""

        return self._call_groq([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.25, max_tokens=512)

    def generate_source_attribution(
        self,
        title: str,
        text: str,
        fact_check_results: List[Dict],
    ) -> Optional[str]:
        """
        Generate a source-attribution summary like:
        "According to Hindustan Times and NDTV, no such incident was reported.
         AltNews rated a similar claim as False on Jan 2024."
        """
        if not fact_check_results and not self.is_available:
            return None

        fc_text = "No fact-checks found for this claim."
        if fact_check_results:
            fc_parts = []
            for fc in fact_check_results[:5]:
                publisher = fc.get('publisher', 'Unknown')
                rating = fc.get('rating', 'Unrated')
                claim = fc.get('claim_text', '')[:150]
                url = fc.get('url', '')
                fc_parts.append(f"- Publisher: {publisher}, Rating: {rating}, Claim: \"{claim}\", URL: {url}")
            fc_text = "\n".join(fc_parts)

        prompt = f"""Based on the following fact-check results, write a 1-2 sentence source attribution summary.
Reference publishers by name. Be specific.

CLAIM: {title}
CONTENT: {text[:300]}

FACT-CHECK RESULTS:
{fc_text}

Write like: "According to [Publisher], [finding]. [Another Publisher] rated this claim as [rating]."
If no fact-checks exist, say "No independent fact-checks were found for this claim as of now."
Plain text only, no markdown."""

        return self._call_groq([
            {"role": "system", "content": "You are a concise fact-check reporter. Write source attributions referencing publishers by name."},
            {"role": "user", "content": prompt},
        ], temperature=0.15, max_tokens=200)
