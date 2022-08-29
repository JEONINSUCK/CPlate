
from django.contrib import admin
from django.urls import path, include
from config.settings import base
# from django.conf.urls import url
# from rest_framework_swagger.views import get_swagger_view

# schema_view = get_swagger_view(title='Pastebin API')

admin.site.site_title = base.ADMIN_SITE_HEADER
admin.site.site_header = base.ADMIN_SITE_HEADER


urlpatterns = [
    path('admin/', admin.site.urls),
    # path('user/', include('user.urls')),
]
