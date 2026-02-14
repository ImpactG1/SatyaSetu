from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # API endpoints
    path('api/analyze/', views.analyze_content_api, name='analyze_content'),
    path('api/analyze-image/', views.analyze_image_api, name='analyze_image'),
    path('api/analyze-audio/', views.analyze_audio_api, name='analyze_audio'),
    path('api/fetch-news/', views.fetch_news_api, name='fetch_news'),
    path('api/forecast/', views.forecast_api, name='forecast'),
    path('api/alerts/', views.get_alerts_api, name='get_alerts'),
    path('api/stats/', views.get_stats_api, name='get_stats'),
    path('api/analysis/<int:analysis_id>/', views.get_analysis_detail_api, name='analysis_detail'),
    path('api/alert/<int:alert_id>/', views.get_alert_detail_api, name='alert_detail'),
]
