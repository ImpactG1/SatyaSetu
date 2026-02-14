from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class Source(models.Model):
    """Information source being monitored"""
    SOURCE_TYPES = [
        ('news', 'News Outlet'),
        ('social', 'Social Media'),
        ('fact_check', 'Fact Check API'),
        ('web', 'Web Source'),
    ]
    
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=500, blank=True, null=True)
    source_type = models.CharField(max_length=20, choices=SOURCE_TYPES)
    credibility_score = models.FloatField(default=5.0)  # 0-10 scale
    is_active = models.BooleanField(default=True)
    last_checked = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.get_source_type_display()})"
    
    class Meta:
        ordering = ['-created_at']


class Content(models.Model):
    """Content item being analyzed"""
    CONTENT_TYPES = [
        ('article', 'News Article'),
        ('claim', 'Claim / Statement'),
        ('social', 'Social Media Post'),
        ('image', 'Image / Screenshot'),
        ('audio', 'Audio Clip'),
        ('other', 'Other'),
    ]
    
    title = models.CharField(max_length=500)
    text = models.TextField()
    url = models.URLField(max_length=1000, blank=True, null=True)
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPES, default='article')
    author = models.CharField(max_length=255, blank=True, null=True)
    published_date = models.DateTimeField(null=True, blank=True)
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name='contents')
    
    # Media file for image/audio uploads
    media_file = models.FileField(upload_to='uploads/%Y/%m/%d/', blank=True, null=True)
    extracted_text = models.TextField(blank=True, default='')  # OCR or transcription output
    extraction_confidence = models.FloatField(default=0.0)  # OCR/STT confidence score
    extraction_metadata = models.JSONField(default=dict, blank=True)  # Additional extraction info
    
    # Aggregated from multiple sources
    related_urls = models.JSONField(default=list, blank=True)  # Links to related content
    images = models.JSONField(default=list, blank=True)  # Image URLs
    
    # Status
    is_analyzed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title[:100]
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['is_analyzed']),
        ]


class MisinformationAnalysis(models.Model):
    """AI Analysis results for content"""
    RISK_LEVELS = [
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
        ('critical', 'Critical Risk'),
    ]
    
    content = models.OneToOneField(Content, on_delete=models.CASCADE, related_name='analysis')
    
    # Detection scores (0-1 scale)
    misinformation_likelihood = models.FloatField(default=0.0)
    credibility_score = models.FloatField(default=0.0)
    bias_score = models.FloatField(default=0.0)
    
    # Prediction metrics
    amplification_risk = models.FloatField(default=0.0)  # Likelihood of going viral
    estimated_reach = models.IntegerField(default=0)  # Estimated audience size
    velocity_score = models.FloatField(default=0.0)  # Speed of spread
    
    # Impact assessment
    societal_impact_score = models.FloatField(default=0.0)  # 0-10 scale
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, default='low')
    affected_topics = models.JSONField(default=list, blank=True)  # Topics/categories affected
    
    # Sentiment & emotional analysis
    sentiment_score = models.FloatField(default=0.0)  # -1 to 1 (negative to positive)
    emotional_triggers = models.JSONField(default=list, blank=True)  # Fear, anger, etc.
    
    # Explainability
    explanation = models.TextField(blank=True)  # Human-readable reasoning
    confidence_score = models.FloatField(default=0.0)  # Model confidence
    key_indicators = models.JSONField(default=list, blank=True)  # Features that triggered detection
    
    # Fact-checking results
    fact_check_results = models.JSONField(default=list, blank=True)  # Results from fact-check APIs
    verified_claims = models.JSONField(default=dict, blank=True)
    
    analyzed_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Analysis for: {self.content.title[:50]}"
    
    class Meta:
        ordering = ['-analyzed_at']
        verbose_name_plural = "Misinformation Analyses"


class Alert(models.Model):
    """System alerts for high-risk content"""
    SEVERITY_LEVELS = [
        ('info', 'Informational'),
        ('warning', 'Warning'),
        ('critical', 'Critical'),
        ('emergency', 'Emergency'),
    ]
    
    analysis = models.ForeignKey(MisinformationAnalysis, on_delete=models.CASCADE, related_name='alerts')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS)
    title = models.CharField(max_length=255)
    message = models.TextField()
    
    # Alert metadata
    impact_areas = models.JSONField(default=list, blank=True)  # Regions, demographics affected
    recommended_actions = models.JSONField(default=list, blank=True)
    
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"[{self.get_severity_display()}] {self.title}"
    
    class Meta:
        ordering = ['-created_at']


class TrendAnalysis(models.Model):
    """Track trends and patterns over time"""
    date = models.DateField()
    topic = models.CharField(max_length=255)
    
    # Aggregated metrics
    total_content_analyzed = models.IntegerField(default=0)
    misinformation_count = models.IntegerField(default=0)
    average_risk_score = models.FloatField(default=0.0)
    high_risk_count = models.IntegerField(default=0)
    
    # Trend data
    trending_keywords = models.JSONField(default=list, blank=True)
    source_distribution = models.JSONField(default=dict, blank=True)
    sentiment_distribution = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.topic} - {self.date}"
    
    class Meta:
        ordering = ['-date']
        unique_together = ['date', 'topic']


class AnalysisLog(models.Model):
    """Audit log for system operations"""
    LOG_TYPES = [
        ('fetch', 'Data Fetch'),
        ('analysis', 'Analysis Run'),
        ('alert', 'Alert Generated'),
        ('error', 'Error'),
    ]
    
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    message = models.TextField()
    details = models.JSONField(default=dict, blank=True)
    success = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"[{self.get_log_type_display()}] {self.created_at}"
    
    class Meta:
        ordering = ['-created_at']
