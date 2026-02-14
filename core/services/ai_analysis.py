"""
AI Analysis Engine
Explainable AI for misinformation detection, risk prediction, and impact assessment
"""

import re
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json

# NLP & Sentiment Analysis
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    # Download required NLTK data (run once)
    # nltk.download('punkt', quiet=True)
    # nltk.download('stopwords', quiet=True)
except ImportError:
    pass

import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class MisinformationDetector:
    """Detect misinformation using multiple heuristics and NLP techniques"""
    
    # Misinformation indicators
    CLICKBAIT_PATTERNS = [
        r'you won\'t believe',
        r'shocking',
        r'doctors hate',
        r'this one trick',
        r'what happens next',
        r'number \d+ will shock you',
        r'the truth about',
        r'they don\'t want you to know',
        r'miracle cure',
        r'!\s*$',  # Excessive exclamation
    ]
    
    SENSATIONAL_WORDS = [
        'breaking', 'urgent', 'exposed', 'revealed', 'secret', 'hidden',
        'conspiracy', 'coverup', 'shocking', 'unbelievable', 'scandal',
        'explosive', 'bombshell', 'exclusive', 'leaked'
    ]
    
    EMOTIONAL_TRIGGER_WORDS = {
        'fear': ['danger', 'threat', 'warning', 'terror', 'deadly', 'fatal', 'disaster'],
        'anger': ['outrage', 'fury', 'rage', 'angry', 'hate', 'disgust'],
        'surprise': ['shocking', 'unbelievable', 'amazing', 'stunning'],
    }
    
    def __init__(self):
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            self.vader = None
            logger.warning("VADER sentiment analyzer not available")
    
    def analyze(self, title: str, text: str, source_credibility: float = 5.0) -> Dict:
        """
        Comprehensive analysis of content for misinformation
        
        Args:
            title: Content title
            text: Content body text
            source_credibility: Source credibility score (0-10)
            
        Returns:
            Dictionary with detection results
        """
        # Initialize results
        results = {
            'misinformation_likelihood': 0.0,
            'credibility_score': 0.0,
            'bias_score': 0.0,
            'key_indicators': [],
            'explanation': '',
            'confidence': 0.0
        }
        
        # Combine title and text for full analysis
        full_text = f"{title} {text}"
        
        # 1. Clickbait detection
        clickbait_score = self._detect_clickbait(title)
        if clickbait_score > 0.5:
            results['key_indicators'].append({
                'type': 'clickbait',
                'score': clickbait_score,
                'description': 'Title contains clickbait patterns'
            })
        
        # 2. Sensationalism detection
        sensational_score = self._detect_sensationalism(full_text)
        if sensational_score > 0.3:
            results['key_indicators'].append({
                'type': 'sensationalism',
                'score': sensational_score,
                'description': 'Content contains sensational language'
            })
        
        # 3. Emotional manipulation
        emotion_scores = self._analyze_emotional_manipulation(full_text)
        if max(emotion_scores.values()) > 0.4:
            results['key_indicators'].append({
                'type': 'emotional_manipulation',
                'scores': emotion_scores,
                'description': 'Content uses emotional trigger words'
            })
        
        # 4. Lack of sources/citations
        citation_score = self._check_citations(text)
        if citation_score < 0.3:
            results['key_indicators'].append({
                'type': 'lack_of_sources',
                'score': citation_score,
                'description': 'Content lacks proper citations or sources'
            })
        
        # 5. Grammar and writing quality
        quality_score = self._assess_writing_quality(text)
        if quality_score < 0.5:
            results['key_indicators'].append({
                'type': 'poor_quality',
                'score': quality_score,
                'description': 'Poor writing quality or excessive errors'
            })
        
        # Calculate overall scores
        misinformation_indicators = [
            clickbait_score * 0.2,
            sensational_score * 0.25,
            max(emotion_scores.values()) * 0.2,
            (1 - citation_score) * 0.2,
            (1 - quality_score) * 0.15
        ]
        
        # Factor in source credibility (inverse relationship)
        source_factor = (10 - source_credibility) / 10
        
        results['misinformation_likelihood'] = min(1.0, np.mean(misinformation_indicators) * (1 + source_factor * 0.5))
        results['credibility_score'] = max(0.0, 1.0 - results['misinformation_likelihood'])
        results['bias_score'] = self._calculate_bias(full_text)
        results['confidence'] = min(0.95, 0.5 + len(results['key_indicators']) * 0.1)
        
        # Generate explanation
        results['explanation'] = self._generate_explanation(results)
        
        return results
    
    def _detect_clickbait(self, title: str) -> float:
        """Detect clickbait patterns in title"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        matches = sum(1 for pattern in self.CLICKBAIT_PATTERNS if re.search(pattern, title_lower))
        
        # Check for excessive punctuation
        exclamation_count = title.count('!')
        question_count = title.count('?')
        
        score = (matches * 0.3) + (min(exclamation_count, 3) * 0.1) + (min(question_count, 2) * 0.05)
        return min(1.0, score)
    
    def _detect_sensationalism(self, text: str) -> float:
        """Detect sensational language"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        sensational_count = sum(1 for word in words if word in self.SENSATIONAL_WORDS)
        score = sensational_count / len(words) * 100  # Normalize
        
        return min(1.0, score)
    
    def _analyze_emotional_manipulation(self, text: str) -> Dict[str, float]:
        """Analyze emotional trigger words"""
        if not text:
            return {emotion: 0.0 for emotion in self.EMOTIONAL_TRIGGER_WORDS}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {emotion: 0.0 for emotion in self.EMOTIONAL_TRIGGER_WORDS}
        
        emotion_scores = {}
        for emotion, trigger_words in self.EMOTIONAL_TRIGGER_WORDS.items():
            count = sum(1 for word in words if word in trigger_words)
            emotion_scores[emotion] = min(1.0, count / len(words) * 100)
        
        return emotion_scores
    
    def _check_citations(self, text: str) -> float:
        """Check for citations and sources"""
        if not text:
            return 0.0
        
        # Look for citation patterns
        citation_patterns = [
            r'according to',
            r'study shows',
            r'research suggests',
            r'expert says',
            r'reported by',
            r'source:',
            r'http[s]?://',
            r'\[\d+\]',  # Reference numbers
        ]
        
        matches = sum(1 for pattern in citation_patterns if re.search(pattern, text.lower()))
        return min(1.0, matches * 0.2)
    
    def _assess_writing_quality(self, text: str) -> float:
        """Assess writing quality"""
        if not text or len(text) < 50:
            return 0.5
        
        # Simple quality heuristics
        sentences = text.split('.')
        if not sentences:
            return 0.5
        
        # Check average sentence length
        avg_sentence_length = len(text) / len(sentences)
        
        # Check for excessive capitals
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
        
        quality = 0.7
        if avg_sentence_length < 10 or avg_sentence_length > 200:
            quality -= 0.2
        if upper_ratio > 0.15:  # Too many capitals
            quality -= 0.3
        
        return max(0.0, quality)
    
    def _calculate_bias(self, text: str) -> float:
        """Calculate potential bias in text"""
        if not text or not self.vader:
            return 0.0
        
        # Use VADER for sentiment
        sentiment = self.vader.polarity_scores(text)
        compound = abs(sentiment['compound'])  # Absolute value - extreme in either direction
        
        return min(1.0, compound)
    
    def _generate_explanation(self, results: Dict) -> str:
        """Generate human-readable explanation"""
        likelihood = results['misinformation_likelihood']
        indicators = results['key_indicators']
        
        if likelihood < 0.3:
            explanation = "Content appears credible with low risk of misinformation. "
        elif likelihood < 0.6:
            explanation = "Content shows moderate signs of misinformation. "
        else:
            explanation = "Content has high likelihood of misinformation. "
        
        if indicators:
            explanation += "Key concerns: "
            concerns = [ind['description'] for ind in indicators[:3]]
            explanation += "; ".join(concerns) + "."
        
        return explanation


class RiskPredictor:
    """Predict amplification risk and societal impact"""
    
    def predict_amplification_risk(self, content_data: Dict, 
                                    misinformation_score: float,
                                    sentiment_score: float) -> Dict:
        """
        Predict likelihood of content going viral
        
        Args:
            content_data: Content metadata
            misinformation_score: Misinformation likelihood score
            sentiment_score: Sentiment analysis score
            
        Returns:
            Dictionary with amplification predictions
        """
        # Factors that increase virality:
        # 1. High emotional content
        # 2. Controversial/shocking nature
        # 3. Simple/shareable message
        # 4. Timing and relevance
        
        emotional_factor = abs(sentiment_score) * 0.3  # Strong emotions increase sharing
        misinformation_factor = misinformation_score * 0.4  # Sensational content spreads faster
        
        # Calculate base amplification risk
        amplification_risk = min(1.0, emotional_factor + misinformation_factor + 0.2)
        
        # Estimate reach based on content characteristics
        base_reach = 1000
        if amplification_risk > 0.7:
            estimated_reach = base_reach * 100  # High risk = potential massive reach
        elif amplification_risk > 0.5:
            estimated_reach = base_reach * 50
        elif amplification_risk > 0.3:
            estimated_reach = base_reach * 10
        else:
            estimated_reach = base_reach
        
        # Velocity score (speed of spread)
        velocity_score = amplification_risk * 0.8
        
        return {
            'amplification_risk': amplification_risk,
            'estimated_reach': int(estimated_reach),
            'velocity_score': velocity_score,
            'explanation': self._explain_amplification(amplification_risk)
        }
    
    def _explain_amplification(self, risk: float) -> str:
        """Explain amplification risk"""
        if risk > 0.7:
            return "High risk of rapid viral spread. Content has characteristics that typically result in massive amplification."
        elif risk > 0.5:
            return "Moderate-to-high amplification risk. Content likely to spread significantly within target communities."
        elif risk > 0.3:
            return "Moderate amplification risk. Content may gain traction but unlikely to go viral."
        else:
            return "Low amplification risk. Limited spread expected."
    
    def assess_societal_impact(self, content_data: Dict,
                               misinformation_score: float,
                               amplification_risk: float,
                               topics: List[str]) -> Dict:
        """
        Assess potential real-world societal impact
        
        Args:
            content_data: Content metadata
            misinformation_score: Misinformation likelihood
            amplification_risk: Predicted amplification risk
            topics: Affected topics/categories
            
        Returns:
            Dictionary with impact assessment
        """
        # High-impact topics
        sensitive_topics = ['health', 'medical', 'vaccine', 'election', 'politics', 
                           'security', 'emergency', 'public safety', 'financial', 'legal']
        
        # Check if content relates to sensitive topics
        topic_sensitivity = 0.0
        affected_topics = []
        for topic in topics:
            if any(sensitive in topic.lower() for sensitive in sensitive_topics):
                topic_sensitivity += 0.3
                affected_topics.append(topic)
        
        topic_sensitivity = min(1.0, topic_sensitivity)
        
        # Calculate impact score (0-10 scale)
        impact_components = [
            misinformation_score * 0.4,  # How misleading is it?
            amplification_risk * 0.3,     # How far will it spread?
            topic_sensitivity * 0.3       # How sensitive is the topic?
        ]
        
        societal_impact_score = sum(impact_components) * 10
        
        # Determine risk level
        if societal_impact_score >= 7.5:
            risk_level = 'critical'
        elif societal_impact_score >= 5.0:
            risk_level = 'high'
        elif societal_impact_score >= 2.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'societal_impact_score': round(societal_impact_score, 2),
            'risk_level': risk_level,
            'affected_topics': affected_topics,
            'explanation': self._explain_impact(societal_impact_score, risk_level, affected_topics)
        }
    
    def _explain_impact(self, score: float, level: str, topics: List[str]) -> str:
        """Explain societal impact"""
        explanation = f"Impact Score: {score}/10 ({level.upper()} risk). "
        
        if level == 'critical':
            explanation += "IMMEDIATE ACTION REQUIRED. Content poses severe threat to public welfare or safety."
        elif level == 'high':
            explanation += "HIGH PRIORITY. Content could significantly influence public opinion or behavior."
        elif level == 'medium':
            explanation += "MONITOR CLOSELY. Content has moderate potential for real-world consequences."
        else:
            explanation += "LOW PRIORITY. Limited real-world impact expected."
        
        if topics:
            explanation += f" Affects sensitive areas: {', '.join(topics)}."
        
        return explanation


class ExplainableAI:
    """Combine all AI components with explainability"""
    
    def __init__(self):
        self.detector = MisinformationDetector()
        self.risk_predictor = RiskPredictor()
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            self.vader = None
    
    def analyze_content(self, title: str, text: str, url: str = '',
                        source_credibility: float = 5.0,
                        topics: List[str] = None) -> Dict:
        """
        Complete explainable AI analysis
        
        Args:
            title: Content title
            text: Content body
            url: Content URL
            source_credibility: Source credibility score (0-10)
            topics: List of topics/categories
            
        Returns:
            Complete analysis results with explanations
        """
        if topics is None:
            topics = self._extract_topics(f"{title} {text}")
        
        # 1. Misinformation Detection
        detection_results = self.detector.analyze(title, text, source_credibility)
        
        # 2. Sentiment Analysis
        sentiment_score = 0.0
        emotional_triggers = []
        if self.vader:
            full_text = f"{title} {text}"
            sentiment = self.vader.polarity_scores(full_text)
            sentiment_score = sentiment['compound']
            
            # Identify emotional triggers
            if sentiment['neg'] > 0.3:
                emotional_triggers.append('negative')
            if sentiment['pos'] > 0.3:
                emotional_triggers.append('positive')
            if abs(sentiment_score) > 0.5:
                emotional_triggers.append('strong_emotion')
        
        # 3. Amplification Risk Prediction
        amplification_results = self.risk_predictor.predict_amplification_risk(
            {'title': title, 'text': text},
            detection_results['misinformation_likelihood'],
            sentiment_score
        )
        
        # 4. Societal Impact Assessment
        impact_results = self.risk_predictor.assess_societal_impact(
            {'title': title, 'text': text},
            detection_results['misinformation_likelihood'],
            amplification_results['amplification_risk'],
            topics
        )
        
        # Combine all results
        complete_analysis = {
            # Detection
            'misinformation_likelihood': detection_results['misinformation_likelihood'],
            'credibility_score': detection_results['credibility_score'],
            'bias_score': detection_results['bias_score'],
            
            # Prediction
            'amplification_risk': amplification_results['amplification_risk'],
            'estimated_reach': amplification_results['estimated_reach'],
            'velocity_score': amplification_results['velocity_score'],
            
            # Impact
            'societal_impact_score': impact_results['societal_impact_score'],
            'risk_level': impact_results['risk_level'],
            'affected_topics': impact_results['affected_topics'],
            
            # Sentiment & Emotion
            'sentiment_score': sentiment_score,
            'emotional_triggers': emotional_triggers,
            
            # Explainability
            'explanation': self._generate_complete_explanation(
                detection_results, amplification_results, impact_results
            ),
            'confidence_score': detection_results['confidence'],
            'key_indicators': detection_results['key_indicators'],
            
            # Fact-checking placeholder (will be filled by API integrations)
            'fact_check_results': [],
            'verified_claims': {}
        }
        
        return complete_analysis
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        # In production, use proper topic modeling or NER
        topics_keywords = {
            'health': ['health', 'medical', 'vaccine', 'disease', 'doctor', 'hospital'],
            'politics': ['election', 'government', 'president', 'congress', 'political'],
            'economy': ['economy', 'financial', 'market', 'stock', 'business'],
            'technology': ['technology', 'ai', 'software', 'computer', 'digital'],
            'science': ['science', 'research', 'study', 'scientific'],
            'social': ['social', 'community', 'society', 'cultural']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topics_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ['general']
    
    def _generate_complete_explanation(self, detection: Dict, 
                                       amplification: Dict, impact: Dict) -> str:
        """Generate comprehensive explanation"""
        parts = [
            f"DETECTION: {detection['explanation']}",
            f"AMPLIFICATION: {amplification['explanation']}",
            f"IMPACT: {impact['explanation']}"
        ]
        
        return " | ".join(parts)
