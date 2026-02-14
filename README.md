# MisInfoShield - Multi-Source Misinformation & Social Risk Early Warning System

An AI-powered system that detects misinformation, predicts viral amplification, and estimates real-world societal impact.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Supabase account

### Installation

1. **Clone and navigate to project:**
   ```powershell
   cd C:\Users\patha\Desktop\hackX
   ```

2. **Activate virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Run the development server:**
   ```powershell
   python manage.py runserver
   ```

4. **Open in browser:**
   ```
   http://127.0.0.1:8000
   ```

## 🔐 Supabase Google OAuth Setup

**IMPORTANT:** To enable Google Sign-In, you need to configure OAuth in Supabase:

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project: `slxysmantzilfkuoofss`
3. Navigate to **Authentication** → **Providers**
4. Enable **Google** provider
5. Add your Google OAuth credentials:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable **Google+ API**
   - Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
   - Application type: **Web application**
   - Add authorized redirect URI:
     ```
     https://slxysmantzilfkuoofss.supabase.co/auth/v1/callback
     ```
   - Copy **Client ID** and **Client Secret** to Supabase

6. In Supabase Authentication Settings, add your site URL:
   ```
   http://127.0.0.1:8000
   ```

7. Add redirect URLs in Supabase:
   ```
   http://127.0.0.1:8000/accounts/callback/
   ```

## 📁 Project Structure

```
hackX/
├── .env                    # Environment variables
├── .gitignore
├── manage.py
├── requirements.txt
├── misinfo_shield/         # Main Django project
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── core/                   # Core app (landing, dashboard)
│   ├── views.py
│   └── urls.py
├── accounts/               # Authentication app
│   ├── views.py
│   └── urls.py
├── templates/              # HTML templates
│   ├── base.html
│   ├── landing.html
│   ├── dashboard.html
│   └── accounts/
│       ├── login.html
│       ├── signup.html
│       └── callback.html
└── static/                 # Static files
    └── css/
        └── style.css
```

## 🎨 Features

- **Beautiful Landing Page** - Modern dark theme with Tailwind CSS
- **Google OAuth** - Sign in with Google via Supabase Auth
- **Email/Password Auth** - Traditional authentication option
- **Dashboard** - Real-time threat monitoring UI
- **PostgreSQL Database** - Hosted on Supabase

## 🛠 Tech Stack

- **Backend:** Django 6.0
- **Frontend:** Tailwind CSS (CDN)
- **Database:** PostgreSQL (Supabase)
- **Authentication:** Supabase Auth (Google OAuth)

## 📦 Dependencies

```
django>=5.0
psycopg2-binary>=2.9
supabase>=2.0
python-dotenv>=1.0
django-cors-headers>=4.0
PyJWT>=2.8
requests>=2.31
```

## 🔗 URLs

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/accounts/login/` | Sign in page |
| `/accounts/signup/` | Sign up page |
| `/dashboard/` | User dashboard (protected) |
| `/admin/` | Django admin |

---

Built for **HackX** - PS-4: Multi-Source Misinformation & Social Risk Early Warning System
