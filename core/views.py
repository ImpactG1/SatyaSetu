from django.shortcuts import render, redirect
from django.conf import settings


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

    return render(request, 'dashboard.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
        'user_email': user_email,
        'user_name': user_name,
        'user_avatar': user_avatar,
        'user_initial': user_initial,
    })
