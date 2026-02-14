from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
import json
import logging

from .models import Content, Source, MisinformationAnalysis, Alert, TrendAnalysis, AnalysisLog
from .services.api_integrations import MultiSourceAggregator, GoogleFactCheckService, NewsAPIService
from .services.ai_analysis import ExplainableAI

logger = logging.getLogger(__name__)


def landing_page(request):
    """Landing page view."""
    return render(request, 'landing.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
    })


def dashboard(request):
    """Dashboard view - requires authentication."""
    if not request.session.get('supabase_user_id'):
        return redirect('accounts:login')

    user_email = request.session.get('user_email', 'User')
    user_name = request.session.get('user_name', user_email.split('@')[0] if '@' in user_email else 'User')
    user_avatar = request.session.get('user_avatar', '')
    user_initial = user_name[0].upper() if user_name else 'U'
    
    # Get dashboard statistics
    try:
        # Sources monitored
        sources_count = Source.objects.filter(is_active=True).count()
        
        # Threats detected (last 7 days)
        seven_days_ago = timezone.now() - timedelta(days=7)
        threats_count = MisinformationAnalysis.objects.filter(
            analyzed_at__gte=seven_days_ago,
            risk_level__in=['high', 'critical']
        ).count()
        
        # Average risk score
        avg_risk = MisinformationAnalysis.objects.filter(
            analyzed_at__gte=seven_days_ago
        ).aggregate(Avg('societal_impact_score'))['societal_impact_score__avg'] or 0.0
        
        # Accuracy rate (placeholder - in production, track true/false positives)
        accuracy_rate = 98.2
        
        # Recent alerts
        recent_alerts = Alert.objects.select_related('analysis__content').order_by('-created_at')[:5]
        
        # Recent analyses
        recent_analyses = MisinformationAnalysis.objects.select_related('content').order_by('-analyzed_at')[:10]
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        sources_count = 0
        threats_count = 0
        avg_risk = 0.0
        accuracy_rate = 0.0
        recent_alerts = []
        recent_analyses = []

    return render(request, 'dashboard.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
        'user_email': user_email,
        'user_name': user_name,
        'user_avatar': user_avatar,
        'user_initial': user_initial,
        'sources_count': sources_count,
        'threats_count': threats_count,
        'avg_risk': round(avg_risk, 1),
        'accuracy_rate': accuracy_rate,
        'recent_alerts': recent_alerts,
        'recent_analyses': recent_analyses,
    })


# API Endpoints

@csrf_exempt
@require_http_methods(["POST"])
def analyze_content_api(request):
    """
    API endpoint to analyze content for misinformation
    POST /api/analyze/
    Body: {
        "title": "Content title",
        "text": "Content body text",
        "url": "https://example.com/article",
        "source_name": "Example Source"
    }
    """
    try:
        data = json.loads(request.body)
        title = data.get('title', '')
        text = data.get('text', '')
        url = data.get('url', '')
        source_name = data.get('source_name', 'Unknown')
        
        if not title or not text:
            return JsonResponse({'error': 'Title and text are required'}, status=400)
        
        # Get or create source
        source, _ = Source.objects.get_or_create(
            name=source_name,
            defaults={'source_type': 'web', 'url': url}
        )
        
        # Create content record
        content = Content.objects.create(
            title=title,
            text=text,
            url=url,
            source=source,
            published_date=timezone.now()
        )
        
        # Run AI analysis
        ai_engine = ExplainableAI()
        analysis_results = ai_engine.analyze_content(
            title=title,
            text=text,
            url=url,
            source_credibility=source.credibility_score
        )
        
        # Get fact-check results
        fact_check_service = GoogleFactCheckService()
        fact_checks = fact_check_service.search_claims(title)
        analysis_results['fact_check_results'] = fact_checks
        
        # Save analysis
        analysis = MisinformationAnalysis.objects.create(
            content=content,
            misinformation_likelihood=analysis_results['misinformation_likelihood'],
            credibility_score=analysis_results['credibility_score'],
            bias_score=analysis_results['bias_score'],
            amplification_risk=analysis_results['amplification_risk'],
            estimated_reach=analysis_results['estimated_reach'],
            velocity_score=analysis_results['velocity_score'],
            societal_impact_score=analysis_results['societal_impact_score'],
            risk_level=analysis_results['risk_level'],
            affected_topics=analysis_results['affected_topics'],
            sentiment_score=analysis_results['sentiment_score'],
            emotional_triggers=analysis_results['emotional_triggers'],
            explanation=analysis_results['explanation'],
            confidence_score=analysis_results['confidence_score'],
            key_indicators=analysis_results['key_indicators'],
            fact_check_results=fact_checks
        )
        
        content.is_analyzed = True
        content.save()
        
        # Create alert if high risk
        if analysis.risk_level in ['high', 'critical']:
            Alert.objects.create(
                analysis=analysis,
                severity='critical' if analysis.risk_level == 'critical' else 'warning',
                title=f"High-risk content detected: {title[:100]}",
                message=analysis.explanation,
                impact_areas=analysis.affected_topics
            )
        
        # Log the analysis
        AnalysisLog.objects.create(
            log_type='analysis',
            message=f"Analyzed content: {title[:100]}",
            details={'content_id': content.id, 'risk_level': analysis.risk_level},
            success=True
        )
        
        return JsonResponse({
            'success': True,
            'content_id': content.id,
            'analysis_id': analysis.id,
            'results': {
                'misinformation_likelihood': analysis.misinformation_likelihood,
                'credibility_score': analysis.credibility_score,
                'risk_level': analysis.risk_level,
                'societal_impact_score': analysis.societal_impact_score,
                'amplification_risk': analysis.amplification_risk,
                'estimated_reach': analysis.estimated_reach,
                'velocity_score': analysis.velocity_score,
                'explanation': analysis.explanation,
                'confidence': analysis.confidence_score,
                'fact_checks': fact_checks
            }
        })
        
    except Exception as e:
        logger.error(f"Error in content analysis: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def fetch_news_api(request):
    """
    Fetch and analyze news from NewsAPI
    POST /api/fetch-news/
    Body: {
        "query": "search term",
        "category": "technology",
        "auto_analyze": true
    }
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        category = data.get('category')
        auto_analyze = data.get('auto_analyze', False)
        
        news_service = NewsAPIService()
        
        if query:
            articles = news_service.search_everything(query)
        elif category:
            articles = news_service.get_top_headlines(category=category)
        else:
            articles = news_service.get_top_headlines()
        
        if not articles:
            return JsonResponse({'error': 'No articles found'}, status=404)
        
        # Store articles
        stored_count = 0
        analyzed_count = 0
        
        for article in articles[:10]:  # Limit to 10 articles
            # Skip articles with missing essential data
            article_title = article.get('title') or ''
            article_text = article.get('content') or article.get('description') or ''
            article_url = article.get('url') or ''
            
            if not article_title or not article_text or not article_url:
                continue
            
            # Get or create source
            source, _ = Source.objects.get_or_create(
                name=article['source'],
                defaults={'source_type': 'news', 'url': article_url}
            )
            
            # Create content
            content, created = Content.objects.get_or_create(
                url=article_url,
                defaults={
                    'title': article_title,
                    'text': article_text,
                    'source': source,
                    'author': article.get('author') or 'Unknown',
                    'published_date': timezone.now()
                }
            )
            
            if created:
                stored_count += 1
                
                # Auto-analyze if requested
                if auto_analyze:
                    try:
                        ai_engine = ExplainableAI()
                        analysis_results = ai_engine.analyze_content(
                            title=content.title,
                            text=content.text,
                            url=content.url,
                            source_credibility=source.credibility_score
                        )
                        
                        MisinformationAnalysis.objects.create(
                            content=content,
                            misinformation_likelihood=analysis_results['misinformation_likelihood'],
                            credibility_score=analysis_results['credibility_score'],
                            bias_score=analysis_results['bias_score'],
                            amplification_risk=analysis_results['amplification_risk'],
                            estimated_reach=analysis_results['estimated_reach'],
                            velocity_score=analysis_results['velocity_score'],
                            societal_impact_score=analysis_results['societal_impact_score'],
                            risk_level=analysis_results['risk_level'],
                            affected_topics=analysis_results['affected_topics'],
                            sentiment_score=analysis_results['sentiment_score'],
                            emotional_triggers=analysis_results['emotional_triggers'],
                            explanation=analysis_results['explanation'],
                            confidence_score=analysis_results['confidence_score'],
                            key_indicators=analysis_results['key_indicators']
                        )
                        
                        content.is_analyzed = True
                        content.save()
                        analyzed_count += 1
                    except Exception as e:
                        logger.error(f"Error analyzing article: {e}")
        
        # Log the fetch
        AnalysisLog.objects.create(
            log_type='fetch',
            message=f"Fetched {len(articles)} articles from NewsAPI",
            details={'stored': stored_count, 'analyzed': analyzed_count},
            success=True
        )
        
        return JsonResponse({
            'success': True,
            'articles_found': len(articles),
            'articles_stored': stored_count,
            'articles_analyzed': analyzed_count
        })
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_alerts_api(request):
    """
    Get recent alerts
    GET /api/alerts/?limit=10&severity=critical
    """
    try:
        limit = int(request.GET.get('limit', 10))
        severity = request.GET.get('severity')
        
        alerts_query = Alert.objects.select_related('analysis__content').order_by('-created_at')
        
        if severity:
            alerts_query = alerts_query.filter(severity=severity)
        
        alerts = alerts_query[:limit]
        
        alerts_data = [{
            'id': alert.id,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'content_title': alert.analysis.content.title,
            'risk_level': alert.analysis.risk_level,
            'impact_score': alert.analysis.societal_impact_score,
            'created_at': alert.created_at.isoformat(),
            'is_acknowledged': alert.is_acknowledged
        } for alert in alerts]
        
        return JsonResponse({'success': True, 'alerts': alerts_data})
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_stats_api(request):
    """
    Get system statistics
    GET /api/stats/
    """
    try:
        seven_days_ago = timezone.now() - timedelta(days=7)
        
        stats = {
            'sources_monitored': Source.objects.filter(is_active=True).count(),
            'total_content': Content.objects.count(),
            'analyzed_content': Content.objects.filter(is_analyzed=True).count(),
            'high_risk_content': MisinformationAnalysis.objects.filter(
                risk_level__in=['high', 'critical']
            ).count(),
            'recent_threats': MisinformationAnalysis.objects.filter(
                analyzed_at__gte=seven_days_ago,
                risk_level__in=['high', 'critical']
            ).count(),
            'avg_risk_score': MisinformationAnalysis.objects.filter(
                analyzed_at__gte=seven_days_ago
            ).aggregate(Avg('societal_impact_score'))['societal_impact_score__avg'] or 0.0,
            'total_alerts': Alert.objects.count(),
            'unacknowledged_alerts': Alert.objects.filter(is_acknowledged=False).count()
        }
        
        return JsonResponse({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return JsonResponse({'error': str(e)}, status=500)
