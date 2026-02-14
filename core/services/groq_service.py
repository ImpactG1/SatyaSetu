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

            if response.status_code != 200:
                logger.error(f"Groq API returned {response.status_code}: {response.text[:300]}")
                return None

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"Groq API response ({len(content)} chars): {content[:100]}...")
            return content

        except requests.Timeout:
            logger.error("Groq API request timed out (30s)")
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
        web_sources: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Generate a professional, detailed AI reasoning explanation.
        Returns human-quality paragraph explaining WHY this content is or isn't misinformation.
        Now includes real web source context from scraping.
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

        # Build web sources context
        web_context = "No web sources scraped."
        if web_sources and web_sources.get('total_sources', 0) > 0:
            web_parts = []
            source_names = web_sources.get('source_names', [])
            consensus = web_sources.get('consensus', 'insufficient')
            web_parts.append(f"  Scraped {web_sources['total_sources']} web sources: {', '.join(source_names[:6])}")
            web_parts.append(f"  Source consensus: {consensus}")

            for ws in web_sources.get('sources_scraped', [])[:4]:
                ws_name = ws.get('source_name', 'Unknown')
                ws_title = ws.get('title', '')[:100]
                ws_snippet = ws.get('snippet', '')[:200]
                ws_type = ws.get('source_type', 'unknown')
                web_parts.append(f"  - {ws_name} ({ws_type}): \"{ws_title}\" — {ws_snippet}")

            web_context = "\n".join(web_parts)

        system_prompt = (
            "You are SatyaSetu AI — a misinformation detection expert. "
            "Generate a concise, professional analysis (3-5 sentences) explaining "
            "your assessment. Reference specific sources by name when available "
            "(e.g., 'According to Hindustan Times...', 'Reuters reports...', "
            "'AltNews found...', 'NDTV coverage shows...'). "
            "Use the WEB SOURCES section to cite real scraped sources by name. "
            "Be specific about what signals raised or lowered suspicion. "
            "Do NOT use markdown. Write in plain text."
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

WEB SOURCES (real-time scraped):
{web_context}

SENSITIVE TOPICS: {topics_text}

Write a clear, professional explanation that:
1. States the overall verdict and confidence
2. References specific web sources and fact-checkers by name (from WEB SOURCES above)
3. Explains the most important red flags or credibility signals
4. Notes what real news organizations are reporting about this topic
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
        web_sources: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Generate a source-attribution summary like:
        "According to Hindustan Times and NDTV, no such incident was reported.
         AltNews rated a similar claim as False on Jan 2024."
        Now uses real web-scraped sources for accurate attribution.
        """
        if not fact_check_results and not web_sources and not self.is_available:
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

        # Build web sources context for attribution
        web_text = "No web sources scraped."
        if web_sources and web_sources.get('total_sources', 0) > 0:
            web_parts = []
            for ws in web_sources.get('sources_scraped', [])[:5]:
                ws_name = ws.get('source_name', 'Unknown')
                ws_title = ws.get('title', '')[:120]
                ws_type = ws.get('source_type', 'unknown')
                ws_url = ws.get('url', '')
                web_parts.append(f"- Source: {ws_name} ({ws_type}), Article: \"{ws_title}\", URL: {ws_url}")
            web_text = "\n".join(web_parts)
            consensus = web_sources.get('consensus', 'insufficient')
            web_text += f"\nOverall consensus: {consensus}"

        prompt = f"""Based on the following fact-check results AND web sources, write a 2-3 sentence source attribution summary.
Reference publishers and news sources by name. Be specific about what each source reports.

CLAIM: {title}
CONTENT: {text[:300]}

FACT-CHECK RESULTS:
{fc_text}

WEB SOURCES (scraped from real websites):
{web_text}

Write like: "According to [Source Name], [what they report]. [Another Source] reports [finding]. [Fact-checker] rated this as [rating]."
Reference as many real sources by name as possible.
If no sources found, say "No independent sources or fact-checks were found for this claim as of now."
Plain text only, no markdown."""

        return self._call_groq([
            {"role": "system", "content": "You are a concise fact-check reporter. Write source attributions referencing publishers by name."},
            {"role": "user", "content": prompt},
        ], temperature=0.15, max_tokens=200)

    def generate_forecast(
        self,
        title: str,
        text: str,
        risk_level: str,
        misinformation_likelihood: float,
        confidence: float,
        affected_topics: List[str],
        web_consensus: str = "insufficient",
        fact_check_count: int = 0,
    ) -> Optional[Dict]:
        """
        Generate predictive forecast scenarios for analyzed content.
        Returns 3 future scenarios with probability percentages.
        """
        pct = round(misinformation_likelihood * 100, 1)
        topics = ", ".join(affected_topics) if affected_topics else "General"

        system_prompt = (
            "You are SatyaSetu AI — a misinformation forecasting expert. "
            "Given an analysis of content, predict 3 plausible future scenarios "
            "with probability percentages that sum to 100%. "
            "Think about viral spread, official responses, fact-checker actions, "
            "platform moderation, and public reaction. "
            "You MUST respond ONLY with valid JSON, no markdown, no code fences."
        )

        user_prompt = f"""Based on this misinformation analysis, predict what will happen next.

CONTENT: {title}
ANALYSIS SUMMARY: {text[:500]}
RISK LEVEL: {risk_level.upper()}
MISINFORMATION LIKELIHOOD: {pct}%
CONFIDENCE: {round(confidence * 100)}%
WEB CONSENSUS: {web_consensus}
FACT-CHECKS FOUND: {fact_check_count}
SENSITIVE TOPICS: {topics}

Generate exactly 3 future scenarios. Respond in this exact JSON format:
{{
    "timeframe": "7-14 days",
    "scenarios": [
        {{
            "title": "Short scenario title (3-6 words)",
            "description": "2-3 sentence description of what happens in this scenario",
            "probability": 45
        }},
        {{
            "title": "Short scenario title",
            "description": "2-3 sentence description",
            "probability": 35
        }},
        {{
            "title": "Short scenario title",
            "description": "2-3 sentence description",
            "probability": 20
        }}
    ],
    "summary": "One sentence overall assessment of the most likely trajectory"
}}

Rules:
- Probabilities must sum to exactly 100
- Scenarios should be distinct and realistic
- Consider the risk level and consensus when assigning probabilities
- If misinformation likelihood is high, weight scenarios toward viral spread and debunking
- If low, weight toward fading from attention or being confirmed
- Be specific and actionable, not generic
- Return ONLY the JSON object, nothing else"""

        raw = self._call_groq([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.4, max_tokens=600)

        if not raw:
            logger.error("Forecast: Groq returned empty response")
            return None

        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            # Handle ```json ... ``` or ``` ... ```
            if cleaned.startswith("```"):
                # Remove opening fence line
                first_newline = cleaned.find("\n")
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            # Handle leading "json" keyword without fence
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            # Find the JSON object boundaries
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]

            result = json.loads(cleaned)

            # Validate structure
            if "scenarios" not in result or not isinstance(result["scenarios"], list):
                logger.error(f"Forecast response missing scenarios array. Keys: {list(result.keys())}")
                return None

            # Ensure each scenario has required fields
            for i, s in enumerate(result["scenarios"]):
                s.setdefault("title", f"Scenario {i+1}")
                s.setdefault("description", "No description available.")
                s.setdefault("probability", 33)

            # Ensure probabilities sum to 100
            total = sum(s["probability"] for s in result["scenarios"])
            if total != 100 and total > 0:
                for s in result["scenarios"]:
                    s["probability"] = round(s["probability"] * 100 / total)

            result.setdefault("timeframe", "7-14 days")
            result.setdefault("summary", "")

            logger.info(f"Forecast parsed OK: {len(result['scenarios'])} scenarios")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse forecast JSON: {e} — raw: {raw[:300]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing forecast: {e}")
            return None
