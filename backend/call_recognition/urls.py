from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter, SimpleRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from .views.user import RegisterUserView, ConsultantViewSet, MeView
from .views.call import StatsView, CallViewSet


call_router = DefaultRouter()
call_router.register("calls", CallViewSet, basename="call")

consultant_router = DefaultRouter()
consultant_router.register("consultants", ConsultantViewSet, basename="consultant")

urlpatterns = [
    path('admin/', admin.site.urls),

    path('register/', RegisterUserView.as_view(), name='register'),
    path('auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/token/verify/', TokenVerifyView.as_view(), name='token_verify'),

    path('me/', MeView.as_view(), name='me'),
    path("", include(consultant_router.urls)),

    path("stats/", StatsView.as_view(), name="stats"),
    path("", include(call_router.urls)),

    # --- OpenAPI 3 & Swagger / Redoc ---
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    path('docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
