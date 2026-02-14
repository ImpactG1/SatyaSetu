from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('callback/', views.callback_view, name='callback'),
    path('auth/callback/', views.auth_callback, name='auth_callback'),
    path('logout/', views.logout_view, name='logout'),
]
