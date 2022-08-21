"""
WSGI config for web project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

MODE = 'dev'

try:
  MODE = os.environ['DJANGO_ENV'].lower()
except:
  pass

print("MODE is {}, {}".format(MODE, __file__))

if MODE == 'prod':
  os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings.prod')
else:
  os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings.dev')

application = get_wsgi_application()
