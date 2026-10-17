from rest_framework import status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from call_recognition.models.call import Call
from call_recognition.utils.permissions import CallPermission
from call_recognition.utils.processing import enqueue_processing, retry_processing
from call_recognition.utils.selectors import calls_for_user, filter_calls
from call_recognition.utils.stats import build_stats
from call_recognition.serializers.call import (
    CallCreateSerializer,
    CallDetailSerializer,
    CallListSerializer,
    CallUpdateSerializer,
)


class CallViewSet(ModelViewSet):
    permission_classes = [CallPermission]
    http_method_names = ["get", "post", "patch", "delete"]

    serializer_classes = {
        "create": CallCreateSerializer,
        "retrieve": CallDetailSerializer,
        "partial_update": CallUpdateSerializer,
    }

    def get_queryset(self):
        qs = filter_calls(calls_for_user(self.request.user), self.request.query_params)
        if self.action == "retrieve":
            qs = qs.select_related("transcript")
        return qs

    def get_serializer_class(self):
        return self.serializer_classes.get(self.action, CallListSerializer)

    def perform_create(self, serializer):
        call = serializer.save(uploaded_by=self.request.user)
        enqueue_processing(call)

    @action(detail=True, methods=["post"])
    def retry(self, request, pk=None):
        """Перезапустить обработку (например, после ошибки)."""
        call = self.get_object()
        if not retry_processing(call):
            return Response({"detail": "Звонок уже обрабатывается"}, status=status.HTTP_409_CONFLICT)
        return Response(CallListSerializer(call).data, status=status.HTTP_202_ACCEPTED)


class StatsView(APIView):
    """Агрегаты для графиков. Принимает те же фильтры, что и список звонков."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = filter_calls(calls_for_user(request.user), request.query_params)
        qs = qs.filter(status=Call.Status.DONE, evaluation__isnull=False)
        return Response(build_stats(qs))