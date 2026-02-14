from django.contrib import admin
from .models import Source, Content, MisinformationAnalysis, Alert, TrendAnalysis, AnalysisLog


@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ('name', 'source_type', 'credibility_score', 'is_active', 'last_checked')
    list_filter = ('source_type', 'is_active')
    search_fields = ('name', 'url')


@admin.register(Content)
class ContentAdmin(admin.ModelAdmin):
    list_display = ('title', 'source', 'is_analyzed', 'created_at')
    list_filter = ('is_analyzed', 'source')
    search_fields = ('title', 'text')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(MisinformationAnalysis)
class MisinformationAnalysisAdmin(admin.ModelAdmin):
    list_display = ('content', 'risk_level', 'misinformation_likelihood', 'societal_impact_score', 'analyzed_at')
    list_filter = ('risk_level',)
    readonly_fields = ('analyzed_at', 'updated_at')


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('title', 'severity', 'is_acknowledged', 'created_at')
    list_filter = ('severity', 'is_acknowledged')
    readonly_fields = ('created_at',)


@admin.register(TrendAnalysis)
class TrendAnalysisAdmin(admin.ModelAdmin):
    list_display = ('topic', 'date', 'total_content_analyzed', 'misinformation_count', 'average_risk_score')
    list_filter = ('date',)


@admin.register(AnalysisLog)
class AnalysisLogAdmin(admin.ModelAdmin):
    list_display = ('log_type', 'message', 'success', 'created_at')
    list_filter = ('log_type', 'success')
    readonly_fields = ('created_at',)
