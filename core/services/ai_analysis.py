"""
AI Analysis Engine v2 — Deep Misinformation Detection
Explainable AI for misinformation detection, risk prediction, and impact assessment.

Key improvements over v1:
- Claim plausibility analysis (detects extraordinary/implausible claims)
- Numerical anomaly detection (flags unrealistic numbers)
- Vague attribution detection ("according to media", "sources say")
- Cross-referencing with fact-check API results
- Semantic topic sensitivity (crime, health, politics get higher base scores)
- Multi-signal fusion with proper weighting
- Much better explanation generation
"""

import re
import math
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

import numpy as np

from .groq_service import GroqReasoningService
from .web_scraper import WebSearchScraper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal 1 – Claim Plausibility Analyzer
# ---------------------------------------------------------------------------

class ClaimPlausibilityAnalyzer:
    """
    Detects implausible, extraordinary, or unverifiable claims.
    This is the MOST important signal — it catches things like
    "500 girls kidnapped in Mumbai and Delhi".
    """

    # Patterns that indicate extraordinary / hard-to-believe claims
    EXTRAORDINARY_PATTERNS = [
        # Mass-casualty / mass-victim claims with large numbers
        (r'\b(\d{2,})\s*(people|persons|children|girls|boys|women|men|students|workers|soldiers|victims|dead|killed|murdered|kidnapped|abducted|missing|injured|infected|died|hospitalized|trapped|displaced|stranded)\b', 'mass_event'),
        (r'\b(kidnapped|abducted|missing|murdered|killed|dead|arrested|raped|assaulted|lynched|trafficked)\b.*?\b(\d{2,})\b', 'mass_event_reverse'),
        (r'\b(\d{2,})\b.*?\b(kidnapped|abducted|missing|murdered|killed|dead|arrested|raped|assaulted|lynched|trafficked)\b', 'mass_event_forward'),
        # Absolute / universal claims
        (r'\b(all|every\s+single|always|never|no\s+one|everyone|100\s*%|completely|totally|entirely|zero\s+cases)\b', 'absolute_claim'),
        # Conspiracy language
        (r'\b(cover[\s-]*up|conspiracy|secret\s*plan|hidden\s*agenda|they\s+don\'?t\s+want|suppressed|censored|silenced|big\s+pharma|deep\s+state|new\s+world\s+order|illuminati|globalist|cabal)\b', 'conspiracy'),
        # Miracle / impossible claims
        (r'\b(cure[sd]?\s+(cancer|aids|hiv|diabetes|covid)|miracle\s+cure|100\s*%\s*(effective|safe|cure)|instantly\s+(cure|heal)|eliminate\s+all)\b', 'miracle_claim'),
        # Doomsday / catastrophe
        (r'\b(end\s+of\s+(the\s+)?world|apocalypse|martial\s+law|civil\s+war|total\s+collapse|mass\s+extinction|world\s+war\s+3)\b', 'doomsday'),
        # Secret knowledge / suppressed truth
        (r'\b(what\s+they\s+hide|government\s+hiding|media\s+won\'?t\s+(tell|show|report)|exposed|exposed!|exposed:)\b', 'suppressed_truth'),
        # Foreign leader / impossible political claims
        (r'\b(putin|xi\s*jinping|kim\s*jong|trump|biden|macron|erdogan|netanyahu|zelensky|sunak|starmer)\b.*?\b(prime\s*minister|president|king|queen|ruler|dictator|chancellor)\s+(of\s+)?(india|china|usa|us|america|russia|pakistan|japan|germany|france|uk|britain|brazil|australia|canada|mexico|nigeria|south\s+africa|iran|iraq|israel|saudi|egypt|turkey|indonesia|bangladesh|sri\s+lanka)\b', 'impossible_political'),
        (r'\b(prime\s*minister|president|king|queen|ruler|dictator|chancellor)\s+(of\s+)?(india|china|usa|us|america|russia|pakistan|japan|germany|france|uk|britain|brazil|australia|canada|mexico|nigeria|south\s+africa)\b.*?\b(putin|xi\s*jinping|kim\s*jong|trump|biden|macron|erdogan|netanyahu|zelensky|sunak|starmer)\b', 'impossible_political'),
        # Nationality mismatch in leadership
        (r'\b(russia|russian|chinese|north\s*korean|american|french|turkish|israeli|british|german|pakistani|japanese|australian|canadian|brazilian|mexican|iranian|iraqi|egyptian|nigerian|saudi)\s+(person|citizen|national|leader|politician)\b.*?\b(prime\s*minister|president|ruler)\s+(of\s+)?(india|china|usa|america|russia|pakistan|japan|uk|france|brazil)\b', 'impossible_political'),
        # Dead person alive / alive person dead (common misinfo patterns)
        (r'\b(died|dead|passed\s+away|killed|assassinated)\b.*?\b(spotted|seen|alive|returns|comeback|appeared|resurfaces)\b', 'impossible_event'),
        (r'\b(alive|spotted|seen)\b.*?\b(actually\s+)?(died|dead|passed\s+away|killed)\b', 'impossible_event'),
    ]

    # Vague / unverifiable attribution patterns
    VAGUE_ATTRIBUTION = [
        r'\baccording\s+to\s+(media|sources?|reports?|some|many|experts?|officials?|insiders?)\b',
        r'\b(sources?\s+say|sources?\s+claim|sources?\s+report|reports?\s+suggest|it\s+is\s+said|people\s+are\s+saying|many\s+are\s+saying)\b',
        r'\b(unnamed\s+source|anonymous\s+source|a\s+source\s+close\s+to|undisclosed|unconfirmed)\b',
        r'\b(some\s+experts?|some\s+scientists?|some\s+doctors?)\s+(say|claim|believe|suggest)\b',
        r'\b(viral|going\s+viral|trending|circulating|forwarded)\b',
        r'\b(whatsapp|forward|received\s+this|share\s+this)\b',
    ]

    CRIME_KEYWORDS = [
        'kidnap', 'kidnapped', 'kidnapping', 'abducted', 'abduction',
        'murder', 'murdered', 'killed', 'dead', 'rape', 'raped',
        'assault', 'assaulted', 'attack', 'attacked', 'bomb', 'bombing',
        'explosion', 'shooting', 'shot', 'stabbing', 'stabbed', 'riot',
        'looting', 'arson', 'missing', 'trafficking', 'trafficked',
        'hostage', 'massacre', 'genocide', 'lynching', 'lynched', 'mob',
    ]

    HEALTH_SCARE_KEYWORDS = [
        'vaccine', 'vaccinated', 'died after', 'death after', 'side effect',
        'causes cancer', 'causes death', 'toxic', 'poison', 'poisoned',
        'microchip', 'infertility', '5g', 'radiation', 'bioweapon',
        'lab leak', 'man-made virus', 'depopulation', 'sterilization',
    ]

    def analyze(self, title: str, text: str) -> Dict:
        full = f"{title} {text}"
        full_lower = full.lower()

        score = 0.0
        indicators = []

        # --- A. Extraordinary claim detection ---
        for pattern, claim_type in self.EXTRAORDINARY_PATTERNS:
            matches = re.findall(pattern, full_lower, re.IGNORECASE)
            if matches:
                if claim_type in ('mass_event', 'mass_event_reverse', 'mass_event_forward'):
                    # Extract all numbers from the full text
                    numbers = [int(n) for n in re.findall(r'\b(\d+)\b', full) if n.isdigit()]
                    large_nums = [n for n in numbers if n >= 10]
                    if large_nums:
                        biggest = max(large_nums)
                        # 10 → 0.3, 50 → 0.65, 100 → 0.8, 200+ → 1.0
                        num_score = min(1.0, 0.2 + biggest / 250)
                        score += num_score * 0.40
                        indicators.append({
                            'type': 'extraordinary_claim',
                            'score': round(num_score, 3),
                            'description': f'Claims mass event involving {biggest}+ people — extraordinary claim requiring strong evidence from official sources'
                        })
                        break  # Don't double-count the same claim
                elif claim_type == 'conspiracy':
                    score += 0.30
                    indicators.append({
                        'type': 'conspiracy_language',
                        'score': 0.75,
                        'description': 'Contains conspiracy-theory language patterns'
                    })
                elif claim_type == 'miracle_claim':
                    score += 0.35
                    indicators.append({
                        'type': 'miracle_claim',
                        'score': 0.85,
                        'description': 'Makes miraculous / scientifically implausible health claims'
                    })
                elif claim_type == 'doomsday':
                    score += 0.25
                    indicators.append({
                        'type': 'doomsday_claim',
                        'score': 0.65,
                        'description': 'Contains doomsday / catastrophe predictions'
                    })
                elif claim_type == 'absolute_claim':
                    score += 0.08
                    indicators.append({
                        'type': 'absolute_language',
                        'score': 0.25,
                        'description': 'Uses absolute language (all, every, never, 100%)'
                    })
                elif claim_type == 'suppressed_truth':
                    score += 0.20
                    indicators.append({
                        'type': 'suppressed_truth',
                        'score': 0.55,
                        'description': 'Claims information is being hidden or suppressed'
                    })
                elif claim_type == 'impossible_political':
                    score += 0.50
                    indicators.append({
                        'type': 'impossible_political',
                        'score': 0.95,
                        'description': 'Claims a foreign leader will lead another country — constitutionally/politically impossible'
                    })
                elif claim_type == 'impossible_event':
                    score += 0.35
                    indicators.append({
                        'type': 'impossible_event',
                        'score': 0.80,
                        'description': 'Contains contradictory life/death claims or physically impossible events'
                    })

        # --- B. Numerical anomaly detection (separate from extraordinary patterns) ---
        numbers_in_text = [int(n) for n in re.findall(r'\b(\d+)\b', full) if n.isdigit()]
        has_crime = any(kw in full_lower for kw in self.CRIME_KEYWORDS)
        has_health = any(kw in full_lower for kw in self.HEALTH_SCARE_KEYWORDS)

        if numbers_in_text and (has_crime or has_health):
            large_numbers = [n for n in numbers_in_text if n >= 15]
            if large_numbers:
                biggest = max(large_numbers)
                # For crime: even 15+ victims in a single event is noteworthy
                num_anomaly = min(1.0, 0.3 + biggest / 150)
                already_flagged = any(i['type'] == 'extraordinary_claim' for i in indicators)
                if not already_flagged:
                    score += num_anomaly * 0.25
                    context = 'crime/violence' if has_crime else 'health scare'
                    indicators.append({
                        'type': 'numerical_anomaly',
                        'score': round(num_anomaly, 3),
                        'description': f'Large number ({biggest}) in {context} context — such claims need official verification'
                    })

        # --- C. Vague attribution ---
        vague_count = sum(1 for p in self.VAGUE_ATTRIBUTION if re.search(p, full_lower))
        if vague_count > 0:
            vague_score = min(1.0, vague_count * 0.25)
            score += vague_score * 0.20
            indicators.append({
                'type': 'vague_attribution',
                'score': round(vague_score, 3),
                'description': f'Uses vague/unverifiable attributions ({vague_count} found, e.g. "according to media") instead of naming specific credible sources'
            })

        # --- D. Thin content (real news articles have substance) ---
        word_count = len(full.split())
        if word_count < 40:
            thinness = min(1.0, (40 - word_count) / 30)
            score += thinness * 0.15
            indicators.append({
                'type': 'thin_content',
                'score': round(thinness, 3),
                'description': f'Very short content ({word_count} words) — credible reports include who, what, when, where, and official statements'
            })

        return {
            'score': min(1.0, score),
            'indicators': indicators,
        }


# ---------------------------------------------------------------------------
# Signal 2 – Linguistic Red-Flag Detector
# ---------------------------------------------------------------------------

class LinguisticAnalyzer:
    """Surface-level linguistic signals — clickbait, sensationalism, emotion."""

    CLICKBAIT_PATTERNS = [
        r'you\s+won\'?t\s+believe', r'what\s+happens\s+next', r'number\s+\d+\s+will\s+shock',
        r'this\s+one\s+trick', r'doctors\s+hate', r'they\s+don\'?t\s+want\s+you\s+to\s+know',
        r'the\s+real\s+truth', r'click\s+here', r'share\s+before\s+(it\'?s?\s+)?deleted',
        r'banned\s+video', r'exposed!', r'must\s+watch', r'must\s+read', r'please\s+share',
    ]

    SENSATIONAL_WORDS = {
        'shocking', 'breaking', 'urgent', 'exposed', 'revealed', 'secret', 'hidden',
        'conspiracy', 'coverup', 'unbelievable', 'scandal', 'explosive', 'bombshell',
        'exclusive', 'leaked', 'horrifying', 'terrifying', 'horrific', 'brutal',
        'devastating', 'alarming', 'outrageous', 'disgusting', 'sickening',
        'frightening', 'gruesome', 'appalling', 'atrocious',
    }

    EMOTIONAL_TRIGGERS = {
        'fear':  {'danger', 'threat', 'warning', 'terror', 'deadly', 'fatal', 'disaster',
                  'panic', 'crisis', 'emergency', 'horrifying', 'scary', 'nightmare'},
        'anger': {'outrage', 'fury', 'rage', 'angry', 'hate', 'disgust', 'betrayal',
                  'corruption', 'scam', 'fraud', 'injustice', 'shameful'},
        'shock': {'shocking', 'unbelievable', 'incredible', 'stunning', 'jaw-dropping',
                  'mind-blowing', 'disturbing', 'heartbreaking'},
    }

    def __init__(self):
        self.vader = None
        if SentimentIntensityAnalyzer is not None:
            try:
                self.vader = SentimentIntensityAnalyzer()
            except Exception:
                pass

    def analyze(self, title: str, text: str) -> Dict:
        full = f"{title} {text}"
        full_lower = full.lower()
        words = full_lower.split()
        word_count = len(words) or 1

        indicators = []
        scores = {}

        # Clickbait
        clickbait_hits = sum(1 for p in self.CLICKBAIT_PATTERNS if re.search(p, full_lower))
        excl = title.count('!')
        title_len = len(title) or 1
        caps_ratio = sum(1 for c in title if c.isupper()) / title_len
        scores['clickbait'] = min(1.0,
            clickbait_hits * 0.25 +
            min(excl, 3) * 0.15 +
            (max(0, caps_ratio - 0.15) * 2.0)
        )
        if scores['clickbait'] > 0.15:
            indicators.append({
                'type': 'clickbait', 'score': round(scores['clickbait'], 3),
                'description': 'Title uses clickbait patterns, excessive punctuation, or ALL-CAPS'
            })

        # Sensationalism
        sensational_hits = sum(1 for w in words if w.strip('.,!?;:"\'-') in self.SENSATIONAL_WORDS)
        scores['sensationalism'] = min(1.0, sensational_hits * 0.12 + (sensational_hits / word_count) * 15)
        if scores['sensationalism'] > 0.1:
            indicators.append({
                'type': 'sensationalism', 'score': round(scores['sensationalism'], 3),
                'description': f'Contains {sensational_hits} sensational/loaded word(s)'
            })

        # Emotional manipulation
        emotion_scores = {}
        for emotion, trigger_set in self.EMOTIONAL_TRIGGERS.items():
            hits = sum(1 for w in words if w.strip('.,!?;:"\'-') in trigger_set)
            emotion_scores[emotion] = min(1.0, hits * 0.18)
        max_emotion = max(emotion_scores.values()) if emotion_scores else 0
        scores['emotion'] = max_emotion
        if max_emotion > 0.1:
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            indicators.append({
                'type': 'emotional_trigger', 'score': round(max_emotion, 3),
                'description': f'Uses language that triggers {top_emotion}'
            })

        # VADER sentiment
        sentiment_compound = 0.0
        neg_score = 0.0
        if self.vader:
            vs = self.vader.polarity_scores(full)
            sentiment_compound = vs['compound']
            neg_score = vs['neg']
            if neg_score > 0.30:
                scores['negative_tone'] = neg_score
                indicators.append({
                    'type': 'negative_tone', 'score': round(neg_score, 3),
                    'description': f'Strongly negative emotional tone ({neg_score:.0%} negative)'
                })

        # ALL-CAPS abuse
        allcaps = [w for w in full.split() if w.isupper() and len(w) > 2]
        if len(allcaps) >= 2:
            cap_score = min(1.0, len(allcaps) * 0.15)
            scores['caps_abuse'] = cap_score
            indicators.append({
                'type': 'caps_abuse', 'score': round(cap_score, 3),
                'description': f'{len(allcaps)} words in ALL CAPS — shouting/alarm pattern'
            })

        combined = (
            scores.get('clickbait', 0) * 0.15 +
            scores.get('sensationalism', 0) * 0.25 +
            scores.get('emotion', 0) * 0.25 +
            scores.get('negative_tone', 0) * 0.20 +
            scores.get('caps_abuse', 0) * 0.15
        )

        return {
            'score': min(1.0, combined),
            'sentiment_compound': sentiment_compound,
            'indicators': indicators,
        }


# ---------------------------------------------------------------------------
# Signal 3 – Source / Citation Quality
# ---------------------------------------------------------------------------

class SourceQualityAnalyzer:
    """Checks whether the content cites credible, verifiable sources."""

    CREDIBLE_PATTERNS = [
        r'(reuters|associated\s+press|ap\s+news|bbc|afp|pti|ani|ians)\s+(report|said|confirmed)',
        r'(official\s+statement|press\s+release|government\s+(said|confirmed)|ministry\s+of)',
        r'(police\s+(said|confirmed|reported|spokesperson)|commissioner|inspector\s+general)',
        r'(published\s+in|peer[\s-]reviewed|doi:|arxiv|journal\s+of)',
        r'https?://[a-zA-Z0-9.-]+\.(gov|edu|org|int)/',
        r'(fir\s+(filed|registered|lodged)|chargesheet|court\s+order)',
        r'(spokesperson|press\s+secretary|official\s+spokesperson)',
        r'(study\s+published|research\s+from|university\s+of)',
    ]

    WEAK_PATTERNS = [
        r'according\s+to\s+(media|sources?|reports?|whatsapp|facebook|twitter|social\s+media|instagram)',
        r'(sources?\s+say|sources?\s+claim|it\s+is\s+said|people\s+say|people\s+are\s+saying)',
        r'(forwarded\s+as\s+received|share\s+this|must\s+read|please\s+share|send\s+to\s+everyone)',
        r'(watch\s+the\s+video|see\s+the\s+proof|in\s+this\s+video)',
        r'(whatsapp|forward|received\s+this|chain\s+message)',
    ]

    def analyze(self, text: str) -> Dict:
        text_lower = text.lower()
        indicators = []

        credible_count = sum(1 for p in self.CREDIBLE_PATTERNS if re.search(p, text_lower))
        weak_count = sum(1 for p in self.WEAK_PATTERNS if re.search(p, text_lower))

        if credible_count == 0 and weak_count == 0:
            score = 0.55
            indicators.append({
                'type': 'no_sources', 'score': 0.55,
                'description': 'No verifiable sources or official references cited'
            })
        elif credible_count == 0 and weak_count > 0:
            score = 0.75
            indicators.append({
                'type': 'weak_sources_only', 'score': 0.75,
                'description': f'Only vague/unverifiable attributions ({weak_count} found), no credible official source cited'
            })
        elif credible_count > 0 and weak_count > credible_count:
            score = 0.35
            indicators.append({
                'type': 'mixed_sources', 'score': 0.35,
                'description': 'Mix of credible and vague sources'
            })
        elif credible_count > 0:
            score = max(0.05, 0.20 - credible_count * 0.05)
            if score <= 0.10:
                indicators.append({
                    'type': 'well_sourced', 'score': round(score, 3),
                    'description': f'Content cites {credible_count} credible/official source(s)'
                })
        else:
            score = 0.40

        return {
            'score': min(1.0, score),
            'credible_count': credible_count,
            'weak_count': weak_count,
            'indicators': indicators,
        }


# ---------------------------------------------------------------------------
# Signal 4 – Fact-Check Cross-Reference
# ---------------------------------------------------------------------------

class FactCheckCrossReferencer:
    """
    Uses results from Google Fact Check API to adjust scores.
    If the claim has been fact-checked as FALSE → huge red flag.
    If fact-checked as TRUE → strong credibility signal.
    """

    FALSE_KEYWORDS = [
        'false', 'fake', 'pants on fire', 'misleading', 'mostly false',
        'incorrect', 'fabricated', 'hoax', 'unproven', 'no evidence',
        'not true', 'baseless', 'debunked', 'manipulated', 'satire',
        'scam', 'rumor', 'rumour', 'altered', 'doctored', 'out of context',
    ]
    TRUE_KEYWORDS = [
        'true', 'correct', 'mostly true', 'verified', 'confirmed', 'accurate',
    ]
    MIXED_KEYWORDS = [
        'half true', 'partly true', 'mixture', 'partly false', 'needs context',
        'missing context', 'unverified', 'mixed',
    ]

    def analyze(self, fact_check_results: List[Dict]) -> Dict:
        if not fact_check_results:
            return {
                'score': 0.30,
                'has_results': False,
                'false_count': 0,
                'true_count': 0,
                'indicators': [{
                    'type': 'no_fact_checks',
                    'score': 0.30,
                    'description': 'No existing fact-checks found for this claim — cannot cross-verify with known fact-checkers'
                }]
            }

        false_count = 0
        true_count = 0
        mixed_count = 0
        indicators = []

        for fc in fact_check_results:
            rating = (fc.get('rating') or '').lower()
            if any(kw in rating for kw in self.FALSE_KEYWORDS):
                false_count += 1
            elif any(kw in rating for kw in self.TRUE_KEYWORDS):
                true_count += 1
            elif any(kw in rating for kw in self.MIXED_KEYWORDS):
                mixed_count += 1

        if false_count > 0:
            score = min(1.0, 0.65 + false_count * 0.12)
            indicators.append({
                'type': 'fact_checked_false',
                'score': round(score, 3),
                'description': f'⚠️ FACT-CHECKED AS FALSE/MISLEADING by {false_count} independent fact-checker(s)!'
            })
        elif mixed_count > 0 and true_count == 0:
            score = 0.45
            indicators.append({
                'type': 'fact_check_mixed',
                'score': 0.45,
                'description': f'Fact-checkers rate this as MIXED / NEEDS CONTEXT ({mixed_count} reviews)'
            })
        elif true_count > 0 and false_count == 0:
            score = max(0.0, 0.10 - true_count * 0.05)
            indicators.append({
                'type': 'fact_checked_true',
                'score': round(score, 3),
                'description': f'✅ Verified as TRUE by {true_count} fact-checker(s)'
            })
        else:
            score = 0.25
            indicators.append({
                'type': 'fact_check_inconclusive',
                'score': 0.25,
                'description': f'Fact-check results inconclusive ({len(fact_check_results)} results reviewed)'
            })

        return {
            'score': min(1.0, score),
            'has_results': True,
            'false_count': false_count,
            'true_count': true_count,
            'indicators': indicators,
        }


# ---------------------------------------------------------------------------
# Signal 5 – Topic Sensitivity Classifier
# ---------------------------------------------------------------------------

class TopicSensitivityClassifier:
    """Classifies content into topics and assesses sensitivity level."""

    TOPIC_MAP = {
        'public_safety': {
            'keywords': [
                'kidnap', 'kidnapped', 'kidnapping', 'abducted', 'abduction',
                'murder', 'murdered', 'killed', 'killing', 'shooting', 'bombing',
                'attack', 'attacked', 'riot', 'violence', 'violent', 'crime',
                'rape', 'raped', 'assault', 'stabbing', 'robbery', 'missing',
                'trafficking', 'hostage', 'terrorist', 'terrorism', 'explosion',
                'mob', 'lynching', 'lynched', 'massacre', 'genocide', 'arson',
            ],
            'sensitivity': 0.95,
            'label': 'Public Safety / Crime',
        },
        'health': {
            'keywords': [
                'vaccine', 'vaccination', 'disease', 'virus', 'pandemic', 'epidemic',
                'hospital', 'doctor', 'medical', 'health', 'cancer', 'drug',
                'medicine', 'treatment', 'cure', 'death toll', 'infected', 'patient',
                'symptom', 'outbreak', 'quarantine', 'WHO', 'CDC',
            ],
            'sensitivity': 0.90,
            'label': 'Health / Medical',
        },
        'politics': {
            'keywords': [
                'election', 'government', 'president', 'prime minister', 'minister',
                'congress', 'parliament', 'political', 'party', 'vote', 'voting',
                'democracy', 'opposition', 'ruling', 'law', 'constitution', 'policy',
                'BJP', 'congress', 'AAP', 'senate', 'democrat', 'republican',
            ],
            'sensitivity': 0.80,
            'label': 'Politics / Governance',
        },
        'communal': {
            'keywords': [
                'hindu', 'muslim', 'christian', 'sikh', 'religious', 'temple',
                'mosque', 'church', 'communal', 'caste', 'ethnic', 'racial',
                'minority', 'majority', 'secular', 'jihad', 'conversion',
                'interfaith', 'sectarian', 'bigotry',
            ],
            'sensitivity': 0.95,
            'label': 'Communal / Religious',
        },
        'financial': {
            'keywords': [
                'stock', 'market', 'crash', 'bank', 'scam', 'fraud', 'ponzi',
                'investment', 'crypto', 'bitcoin', 'economy', 'recession',
                'inflation', 'bankruptcy', 'demonetization',
            ],
            'sensitivity': 0.75,
            'label': 'Financial / Economic',
        },
        'disaster': {
            'keywords': [
                'earthquake', 'flood', 'tsunami', 'cyclone', 'hurricane',
                'wildfire', 'drought', 'famine', 'volcano', 'landslide',
                'storm', 'disaster', 'devastation', 'calamity',
            ],
            'sensitivity': 0.85,
            'label': 'Natural Disaster',
        },
        'children': {
            'keywords': [
                'child', 'children', 'minor', 'minors', 'girl', 'girls',
                'boy', 'boys', 'student', 'students', 'school', 'baby',
                'infant', 'toddler', 'teenager', 'juvenile', 'underage',
            ],
            'sensitivity': 0.90,
            'label': 'Children / Minors',
        },
    }

    def classify(self, text: str) -> Dict:
        text_lower = text.lower()
        detected = []
        max_sensitivity = 0.0

        for topic_id, topic_info in self.TOPIC_MAP.items():
            hits = sum(1 for kw in topic_info['keywords'] if kw in text_lower)
            if hits > 0:
                detected.append({
                    'id': topic_id,
                    'label': topic_info['label'],
                    'sensitivity': topic_info['sensitivity'],
                    'keyword_hits': hits,
                })
                max_sensitivity = max(max_sensitivity, topic_info['sensitivity'])

        return {
            'topics': detected,
            'labels': [t['label'] for t in detected],
            'max_sensitivity': max_sensitivity,
            'is_sensitive': max_sensitivity >= 0.80,
        }


# ---------------------------------------------------------------------------
# MASTER: Explainable AI Engine  (multi-signal fusion)
# ---------------------------------------------------------------------------

class ExplainableAI:
    """
    Fuses all 5 signals with proper weighting to produce a final verdict.

    Signal weights:
      Claim Plausibility  : 0.30  — Is the claim itself believable?
      Linguistic Red Flags : 0.10  — Clickbait, sensationalism, emotion
      Source Quality       : 0.20  — Are credible sources cited?
      Fact-Check X-Ref     : 0.25  — Has this been fact-checked as false/true?
      Topic Sensitivity    : 0.15  — How dangerous/sensitive is the topic?
    """

    SIGNAL_WEIGHTS = {
        'plausibility':  0.35,
        'linguistic':    0.08,
        'source':        0.18,
        'fact_check':    0.25,
        'topic':         0.14,
    }

    def __init__(self):
        self.plausibility = ClaimPlausibilityAnalyzer()
        self.linguistic = LinguisticAnalyzer()
        self.source_quality = SourceQualityAnalyzer()
        self.fact_checker = FactCheckCrossReferencer()
        self.topic_classifier = TopicSensitivityClassifier()
        self.groq = GroqReasoningService()
        self.web_scraper = WebSearchScraper()

    def analyze_content(self, title: str, text: str, url: str = '',
                        source_credibility: float = 5.0,
                        topics: List[str] = None,
                        fact_check_results: List[Dict] = None,
                        web_sources: Dict = None) -> Dict:
        """
        Complete explainable AI analysis with multi-signal fusion.
        Now includes web scraping for real-time source verification.
        """
        full_text = f"{title} {text}"

        # ---- Web scraping for real source verification ----
        if web_sources is None:
            try:
                web_sources = self.web_scraper.search_and_scrape(title)
                logger.info(f"Web scraper found {web_sources.get('total_sources', 0)} sources")
            except Exception as e:
                logger.error(f"Web scraping failed, continuing without: {e}")
                web_sources = {'sources_scraped': [], 'total_sources': 0,
                               'source_names': [], 'consensus': 'insufficient',
                               'summary': 'Web scraping unavailable.'}

        # ---- Run all 5 signal analyzers ----
        plaus = self.plausibility.analyze(title, text)
        ling = self.linguistic.analyze(title, text)
        source = self.source_quality.analyze(text)
        fc = self.fact_checker.analyze(fact_check_results or [])
        topic_info = self.topic_classifier.classify(full_text)

        # ---- LLM plausibility check (catches semantic absurdity regex can't) ----
        llm_plaus = None
        if self.groq.is_available:
            try:
                llm_plaus = self.groq.assess_claim_plausibility(title, text)
                if llm_plaus and llm_plaus.get('score', 0) > 0:
                    logger.info(f"LLM plausibility: {llm_plaus['score']:.2f} — {llm_plaus.get('reason', '')[:80]}")
            except Exception as e:
                logger.error(f"LLM plausibility check failed: {e}")

        # Merge LLM plausibility with regex plausibility — take the HIGHER score
        if llm_plaus and llm_plaus.get('score', 0) > plaus['score']:
            llm_score = llm_plaus['score']
            plaus['score'] = llm_score
            plaus['indicators'].append({
                'type': 'llm_implausible',
                'score': round(llm_score, 3),
                'description': f'AI plausibility check: {llm_plaus.get("reason", "Claim appears implausible")}'
            })

        # ---- Collect all indicators ----
        all_indicators = (
            plaus['indicators'] +
            ling['indicators'] +
            source['indicators'] +
            fc['indicators']
        )

        # ---- Multi-signal fusion ----
        raw_scores = {
            'plausibility': plaus['score'],
            'linguistic': ling['score'],
            'source': source['score'],
            'fact_check': fc['score'],
            'topic': topic_info['max_sensitivity'],
        }

        # Weighted sum
        weighted_sum = sum(raw_scores[k] * self.SIGNAL_WEIGHTS[k] for k in self.SIGNAL_WEIGHTS)

        # BOOST: if plausibility flags are high AND sources are weak → compound
        if plaus['score'] >= 0.4 and source['score'] >= 0.45:
            weighted_sum = min(1.0, weighted_sum * 1.35)

        # BOOST: sensitive topic + any plausibility concern → amplify
        if topic_info['is_sensitive'] and plaus['score'] >= 0.3:
            weighted_sum = min(1.0, weighted_sum * 1.25)

        # BOOST: fact-checked as false → ensure minimum score of 0.80
        if fc.get('false_count', 0) > 0:
            weighted_sum = max(weighted_sum, 0.80)

        # PENALTY: low source credibility
        source_penalty = max(0, (5.0 - source_credibility) / 10.0) * 0.12
        weighted_sum = min(1.0, weighted_sum + source_penalty)

        # FLOOR: if we have extraordinary claims + no credible sources → minimum 0.55
        if plaus['score'] >= 0.35 and source['score'] >= 0.50:
            weighted_sum = max(weighted_sum, 0.55)

        # LLM OVERRIDE: if LLM is highly confident the claim is absurd → enforce floor
        if llm_plaus and llm_plaus.get('score', 0) >= 0.85:
            weighted_sum = max(weighted_sum, 0.82)
        elif llm_plaus and llm_plaus.get('score', 0) >= 0.70:
            weighted_sum = max(weighted_sum, 0.68)
        elif llm_plaus and llm_plaus.get('score', 0) >= 0.55:
            weighted_sum = max(weighted_sum, 0.55)

        # ---- Web source consensus adjustment ----
        web_consensus = web_sources.get('consensus', 'insufficient')
        web_source_count = web_sources.get('total_sources', 0)

        if web_consensus == 'mostly_denied' and web_source_count >= 2:
            # Multiple sources deny the claim → boost misinfo score
            weighted_sum = max(weighted_sum, 0.70)
            all_indicators.append({
                'type': 'web_sources_deny',
                'score': 0.85,
                'description': f'Web sources ({web_source_count} scraped) mostly deny or debunk this claim'
            })
        elif web_consensus == 'mostly_supported' and web_source_count >= 2:
            # Sources support the claim → lower misinfo score
            weighted_sum = min(weighted_sum, weighted_sum * 0.65)
            all_indicators.append({
                'type': 'web_sources_support',
                'score': 0.15,
                'description': f'Web sources ({web_source_count} scraped) generally support this claim'
            })
        elif web_consensus == 'conflicting':
            all_indicators.append({
                'type': 'web_sources_conflicting',
                'score': 0.50,
                'description': f'Web sources show conflicting information about this claim'
            })

        # Boost from fact-checker websites in scraped sources
        fc_web_count = web_sources.get('fact_checker_sources', 0)
        if fc_web_count > 0:
            # Check if fact-checkers found on web also flag it
            for ws in web_sources.get('sources_scraped', []):
                if ws.get('source_type') == 'fact_checker':
                    ws_text = (ws.get('full_text', '') + ws.get('title', '')).lower()
                    deny_words = ['false', 'fake', 'hoax', 'misleading', 'debunked', 'not true']
                    if any(w in ws_text for w in deny_words):
                        weighted_sum = max(weighted_sum, 0.80)
                        all_indicators.append({
                            'type': 'fact_checker_website_deny',
                            'score': 0.90,
                            'description': f'{ws["source_name"]} (fact-checker) flags this claim as false/misleading'
                        })
                        break

        misinformation_likelihood = round(min(1.0, weighted_sum), 4)
        credibility_score = round(1.0 - misinformation_likelihood, 4)

        # ---- Sentiment ----
        sentiment_compound = ling.get('sentiment_compound', 0.0)
        emotional_triggers = []
        if sentiment_compound < -0.3:
            emotional_triggers.append('negative')
        if sentiment_compound > 0.3:
            emotional_triggers.append('positive')
        if abs(sentiment_compound) > 0.5:
            emotional_triggers.append('strong_emotion')

        # ---- Amplification Risk ----
        amp = self._predict_amplification(
            misinformation_likelihood, sentiment_compound,
            topic_info, plaus['score']
        )

        # ---- Societal Impact ----
        impact = self._assess_impact(
            misinformation_likelihood, amp['amplification_risk'],
            topic_info
        )

        # ---- Confidence ----
        indicator_count = len(all_indicators)
        confidence = min(0.95, 0.40 + indicator_count * 0.06 +
                         (0.12 if fc.get('has_results') else 0))

        # ---- Explanation (Groq LLM if available, else fallback) ----
        groq_explanation = None
        source_attribution = None
        if self.groq.is_available:
            try:
                groq_explanation = self.groq.generate_deep_reasoning(
                    title=title, text=text,
                    signal_scores=raw_scores,
                    key_indicators=all_indicators,
                    fact_check_results=fact_check_results or [],
                    topic_info={'labels': topic_info['labels']},
                    misinformation_likelihood=misinformation_likelihood,
                    risk_level=impact['level'],
                    web_sources=web_sources,
                )
                source_attribution = self.groq.generate_source_attribution(
                    title=title, text=text,
                    fact_check_results=fact_check_results or [],
                    web_sources=web_sources,
                )
            except Exception as e:
                logger.error(f"Groq reasoning failed, using fallback: {e}")

        # Fallback explanation if Groq unavailable
        fallback_explanation = self._build_explanation(
            misinformation_likelihood, raw_scores, all_indicators,
            amp, impact, topic_info
        )
        explanation = groq_explanation or fallback_explanation

        # ---- Bias score ----
        bias_score = abs(sentiment_compound)

        return {
            'misinformation_likelihood': misinformation_likelihood,
            'credibility_score': credibility_score,
            'bias_score': round(bias_score, 4),

            'amplification_risk': amp['amplification_risk'],
            'estimated_reach': amp['estimated_reach'],
            'velocity_score': amp['velocity_score'],

            'societal_impact_score': impact['score'],
            'risk_level': impact['level'],
            'affected_topics': topic_info['labels'] or ['general'],

            'sentiment_score': sentiment_compound,
            'emotional_triggers': emotional_triggers,

            'explanation': explanation,
            'confidence_score': round(confidence, 4),
            'key_indicators': all_indicators,

            'fact_check_results': fact_check_results or [],
            'verified_claims': {},
            'source_attribution': source_attribution or '',
            'signal_scores': raw_scores,

            # Web scraping results
            'web_sources': {
                'total': web_sources.get('total_sources', 0),
                'source_names': web_sources.get('source_names', []),
                'consensus': web_sources.get('consensus', 'insufficient'),
                'summary': web_sources.get('summary', ''),
                'fact_checker_count': web_sources.get('fact_checker_sources', 0),
                'mainstream_count': web_sources.get('mainstream_sources', 0),
                'sources_detail': [
                    {
                        'name': s.get('source_name', 'Unknown'),
                        'domain': s.get('source_domain', ''),
                        'type': s.get('source_type', 'unknown'),
                        'credibility': s.get('credibility', 5.0),
                        'title': s.get('title', ''),
                        'snippet': s.get('snippet', '')[:200],
                        'url': s.get('url', ''),
                        'relevance': s.get('relevance_score', 0),
                    }
                    for s in web_sources.get('sources_scraped', [])[:6]
                ],
            },
        }

    # --- Amplification predictor ---

    def _predict_amplification(self, misinfo_score: float, sentiment: float,
                                topic_info: Dict, plausibility_score: float) -> Dict:
        emotional_factor = abs(sentiment) * 0.20
        misinfo_factor = misinfo_score * 0.30
        topic_factor = topic_info['max_sensitivity'] * 0.30
        plaus_factor = plausibility_score * 0.20

        risk = min(1.0, emotional_factor + misinfo_factor + topic_factor + plaus_factor)

        # Sensitive topics + misinformation = viral amplification
        if topic_info['is_sensitive'] and misinfo_score > 0.4:
            risk = min(1.0, risk * 1.4)

        if risk >= 0.75:
            reach = 500_000
        elif risk >= 0.55:
            reach = 100_000
        elif risk >= 0.35:
            reach = 25_000
        else:
            reach = 5_000

        velocity = round(risk * 0.85, 4)

        if risk > 0.7:
            expl = "VERY HIGH risk of rapid viral spread. Content touches sensitive topics and has characteristics that drive mass sharing on social media and messaging apps."
        elif risk > 0.5:
            expl = "HIGH amplification risk. Likely to spread significantly in communities and social platforms."
        elif risk > 0.3:
            expl = "MODERATE amplification risk. May gain some traction but unlikely to go massively viral."
        else:
            expl = "LOW amplification risk. Limited organic spread expected."

        return {
            'amplification_risk': round(risk, 4),
            'estimated_reach': reach,
            'velocity_score': velocity,
            'explanation': expl,
        }

    # --- Impact assessor ---

    def _assess_impact(self, misinfo_score: float, amp_risk: float,
                        topic_info: Dict) -> Dict:

        topic_weight = topic_info['max_sensitivity']

        raw = (misinfo_score * 0.40 + amp_risk * 0.30 + topic_weight * 0.30) * 10
        score = round(min(10.0, raw), 2)

        if score >= 7.5:
            level = 'critical'
        elif score >= 5.0:
            level = 'high'
        elif score >= 2.5:
            level = 'medium'
        else:
            level = 'low'

        return {'score': score, 'level': level}

    # --- Explanation builder ---

    def _build_explanation(self, likelihood: float, signals: Dict,
                           indicators: List[Dict], amp: Dict, impact: Dict,
                           topic_info: Dict) -> str:
        parts = []

        # Overall verdict
        pct = likelihood * 100
        if likelihood >= 0.7:
            parts.append(f"⚠️ HIGH RISK — {pct:.0f}% misinformation likelihood.")
        elif likelihood >= 0.4:
            parts.append(f"⚡ MODERATE RISK — {pct:.0f}% misinformation likelihood.")
        else:
            parts.append(f"✅ LOW RISK — {pct:.0f}% misinformation likelihood.")

        # Top concerns (sorted by score)
        top = sorted(indicators, key=lambda x: x.get('score', 0), reverse=True)[:4]
        if top:
            parts.append("KEY CONCERNS: " + " • ".join(i['description'] for i in top))

        # Topics
        if topic_info['labels']:
            parts.append(f"SENSITIVE TOPICS: {', '.join(topic_info['labels'])}.")

        # Amplification
        parts.append(f"SPREAD: {amp['explanation']}")

        # Impact
        parts.append(f"SOCIETAL IMPACT: {impact['score']}/10 ({impact['level'].upper()}).")

        return " | ".join(parts)
