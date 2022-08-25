from pathlib import Path
import os
from unittest.mock import DEFAULT

BASE_DIR = Path(__file__).resolve().parent.parent

# admin site
ADMIN_SITE_HEADER = 'CPlate CMS Center'

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'kiss,dry,yagni'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS =  ['localhost', '127.0.0.1']

ALLOWED_HOSTS = ['*']
ROOT_HOSTCONF = 'url'
DEFAULT_HOST = 'www'

# Application definition
# 프로젝트에서 사용할 앱들의 경로가 위치하는 영역, user defined app의 경로도 이 항목에 포함된다.
INSTALLED_APPS = [
    # 'common.apps.CommonConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # custom
    'drf_yasg',
    'rest_framework',
    # 'django_hosts'
]

# 공통적으로 들어가는 html 코드를 관리하기 위한 확장형 template 들의 경로를 설정하는 영역
MIDDLEWARE = [
    # 'django_hosts.middleware.HostsRequestMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # 'django_hosts.middleware.HostsResponseMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'



# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'ko-kr'
# django의 DateTime 객체에서 사용할 기준 시간을 설정
TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

# DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


AUTH_USER_MODEL = 'user.User'