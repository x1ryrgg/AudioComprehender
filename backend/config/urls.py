from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from call_recognition.views.user import check_health

urlpatterns = [
    path('health/', check_health, name='health'),

    path("api/v1/", include("call_recognition.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
