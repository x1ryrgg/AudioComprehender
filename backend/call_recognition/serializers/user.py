from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from call_recognition.models.user import User


class RegisterUserSerializer(serializers.ModelSerializer):
    confirm_password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "email",
            "password",
            "confirm_password",
            "first_name",
            "last_name",
        )
        extra_kwargs = {"id": {"read_only": True}, "password": {"write_only": True}}

    def validate(self, attrs):
        if attrs.get("password") != attrs.get("confirm_password"):
            raise ValidationError({"confirm_password": "Passwords do not match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop("confirm_password")

        user = User.objects.create_user(**validated_data,)
        return user


class UserSerializer(serializers.ModelSerializer):
    """Профиль текущего пользователя. """

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "role"]
        read_only_fields = ["id", "username", "role"]


class ConsultantSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source="display_name", read_only=True)
    calls_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = User
        fields = ["id", "name", "calls_count"]
