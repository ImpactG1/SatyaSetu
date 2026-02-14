"""
Management command to initialize the system with sample sources
"""
from django.core.management.base import BaseCommand
from core.models import Source


class Command(BaseCommand):
    help = 'Initialize system with default news sources'

    def handle(self, *args, **options):
        sources = [
            {'name': 'BBC News', 'url': 'https://www.bbc.com', 'source_type': 'news', 'credibility_score': 8.5},
            {'name': 'Reuters', 'url': 'https://www.reuters.com', 'source_type': 'news', 'credibility_score': 9.0},
            {'name': 'AP News', 'url': 'https://apnews.com', 'source_type': 'news', 'credibility_score': 9.0},
            {'name': 'CNN', 'url': 'https://www.cnn.com', 'source_type': 'news', 'credibility_score': 7.0},
            {'name': 'The Guardian', 'url': 'https://www.theguardian.com', 'source_type': 'news', 'credibility_score': 7.5},
            {'name': 'Google Fact Check', 'url': 'https://toolbox.google.com/factcheck', 'source_type': 'fact_check', 'credibility_score': 9.5},
            {'name': 'Snopes', 'url': 'https://www.snopes.com', 'source_type': 'fact_check', 'credibility_score': 8.5},
            {'name': 'FactCheck.org', 'url': 'https://www.factcheck.org', 'source_type': 'fact_check', 'credibility_score': 9.0},
        ]
        
        created_count = 0
        for source_data in sources:
            source, created = Source.objects.get_or_create(
                name=source_data['name'],
                defaults=source_data
            )
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'Created source: {source.name}'))
        
        self.stdout.write(self.style.SUCCESS(f'\nInitialization complete! Created {created_count} new sources.'))
        self.stdout.write(f'Total active sources: {Source.objects.filter(is_active=True).count()}')
