from rest_framework.generics import CreateAPIView

from rest_framework import permissions
from call_recognition.serializers.user import UserSerializer

from call_recognition.models import User



class RegisterUserView(CreateAPIView):
    """ Регистрация. """

    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]




