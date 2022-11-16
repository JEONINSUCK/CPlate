from pathlib import Path
import os
from unittest.mock import DEFAULT
from .base import *

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases
# database를 사용하기 위한 설정, default로 sqllite를 사용한다.
DATABASES = {
    'default': {
        # TODO : 나중에 배포 및 개발 설정 파일 분리 해야 함
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'cplate',
        'USER': 'cplate',
        'PASSWORD': 'cplate_password!0C',
        'HOST': 'donghwa.dev',
        'PORT': 65200,
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': BASE_DIR / 'db.sqlite3',
    }
}