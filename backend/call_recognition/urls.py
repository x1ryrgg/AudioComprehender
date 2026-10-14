from django.contrib import admin
from django.urls import path, include
from .views import UnAuthorizedTestView, AuthorizedTestView

urlpatterns = [
    path('authorized/', AuthorizedTestView.as_view()),
    path('unauthorized/', UnAuthorizedTestView.as_view()),

]
