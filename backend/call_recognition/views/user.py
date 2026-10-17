from django.db.models import Count
from django.http import JsonResponse
from rest_framework.generics import CreateAPIView, RetrieveUpdateAPIView

from rest_framework import permissions
from rest_framework.viewsets import ReadOnlyModelViewSet
from call_recognition.utils.permissions import IsSupervisor
from call_recognition.serializers.user import RegisterUserSerializer, UserSerializer, ConsultantSerializer

from call_recognition.models import User


def check_health(request):
    return JsonResponse({"status": "GOOD"})


class RegisterUserView(CreateAPIView):
    """ Регистрация. """

    queryset = User.objects.all()
    serializer_class = RegisterUserSerializer
    permission_classes = [permissions.AllowAny]


class MeView(RetrieveUpdateAPIView):
    """GET /api/me/ — профиль текущего пользователя, PATCH — изменить имя/фамилию/email."""

    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    http_method_names = ["get", "patch"]

    def get_object(self):
        return self.request.user


class ConsultantViewSet(ReadOnlyModelViewSet):
    """Список консультантов для руководителя: выбор консультанта при загрузке звонка, фильтры."""

    serializer_class = ConsultantSerializer
    permission_classes = [IsSupervisor]

    def get_queryset(self):
        return (
            User.objects.filter(role=User.UserRole.CONSULTANT, is_active=True)
            .annotate(calls_count=Count("calls"))
            .order_by("last_name", "first_name", "username")
        )




