from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json


def login_view(request):
    """Login page view."""
    if request.session.get('supabase_user_id'):
        return redirect('core:dashboard')
    
    return render(request, 'accounts/login.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
    })


def signup_view(request):
    """Signup page view."""
    if request.session.get('supabase_user_id'):
        return redirect('core:dashboard')
    
    return render(request, 'accounts/signup.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
    })


def callback_view(request):
    """OAuth callback page - handles the redirect from Supabase OAuth."""
    return render(request, 'accounts/callback.html', {
        'supabase_url': settings.SUPABASE_URL,
        'supabase_anon_key': settings.SUPABASE_ANON_KEY,
    })


@csrf_exempt
@require_http_methods(["POST"])
def auth_callback(request):
    """API endpoint to handle auth session from frontend."""
    try:
        data = json.loads(request.body)
        access_token = data.get('access_token')
        refresh_token = data.get('refresh_token')
        user = data.get('user', {})

        if not access_token or not user:
            return JsonResponse({'success': False, 'error': 'Invalid data'}, status=400)

        # Store user info in session
        request.session['supabase_user_id'] = user.get('id')
        request.session['supabase_access_token'] = access_token
        request.session['supabase_refresh_token'] = refresh_token
        request.session['user_email'] = user.get('email', '')
        
        # Get user metadata
        user_metadata = user.get('user_metadata', {})
        request.session['user_name'] = user_metadata.get('full_name') or user_metadata.get('name', '')
        request.session['user_avatar'] = user_metadata.get('avatar_url') or user_metadata.get('picture', '')

        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def logout_view(request):
    """Logout view - clears session."""
    request.session.flush()
    return redirect('core:landing')
