import os

from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from call_recognition.models.call import Call, Transcript, Evaluation
from call_recognition.models.user import User

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
MAX_SIZE_MB = 100


def _consultants():
    return User.objects.filter(role=User.UserRole.CONSULTANT, is_active=True)


class UserBriefSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source="display_name", read_only=True)

    class Meta:
        model = User
        fields = ["id", "name"]


class CallCreateSerializer(serializers.ModelSerializer):
    consultant = serializers.PrimaryKeyRelatedField(queryset=_consultants(), required=False)

    class Meta:
        model = Call
        fields = ["id", "audio_file", "consultant", "called_at", "status"]
        read_only_fields = ["id", "status"]

    def validate_audio_file(self, f):
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Неподдерживаемый формат. Разрешены: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        if f.size > MAX_SIZE_MB * 1024 * 1024:
            raise serializers.ValidationError(f"Файл больше {MAX_SIZE_MB} МБ")
        return f

    def validate(self, attrs):
        user = self.context["request"].user
        consultant = attrs.get("consultant")
        if user.is_supervisor:
            if consultant is None:
                raise serializers.ValidationError({"consultant": "Укажите консультанта"})
        else:
            # консультант загружает только свои звонки
            if consultant is not None and consultant != user:
                raise serializers.ValidationError({"consultant": "Можно загружать только свои звонки"})
            attrs["consultant"] = user
        return attrs

    def create(self, validated_data):
        validated_data["original_filename"] = validated_data["audio_file"].name
        return super().create(validated_data)


class CallUpdateSerializer(serializers.ModelSerializer):
    """PATCH: менять можно только консультанта и дату звонка (статус, ошибку и т.п. нельзя)."""

    consultant = serializers.PrimaryKeyRelatedField(queryset=_consultants(), required=False)

    class Meta:
        model = Call
        fields = ["consultant", "called_at"]


class CallListSerializer(serializers.ModelSerializer):
    consultant = UserBriefSerializer(read_only=True)
    final_score = serializers.FloatField(source="evaluation.final_score", read_only=True, allow_null=True)
    text_score = serializers.FloatField(source="evaluation.text_score", read_only=True, allow_null=True)
    emotion_score = serializers.FloatField(source="evaluation.emotion_score", read_only=True, allow_null=True)
    flags = serializers.JSONField(source="evaluation.flags", read_only=True, allow_null=True)

    class Meta:
        model = Call
        fields = [
            "id", "original_filename", "consultant", "called_at", "duration",
            "status", "error", "created_at", "processed_at",
            "final_score", "text_score", "emotion_score", "flags",
        ]


class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = [
            "final_score", "text_score", "emotion_score", "negative_share",
            "flags", "text_analysis", "emotion_analysis",
        ]


class TranscriptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcript
        fields = ["segments"]


class CallDetailSerializer(serializers.ModelSerializer):
    consultant = UserBriefSerializer(read_only=True)
    # Пока звонок не обработан, этих блоков в ответе просто не будет
    evaluation = EvaluationSerializer(read_only=True)
    transcript = TranscriptSerializer(read_only=True)

    class Meta:
        model = Call
        fields = [
            "id", "audio_file", "original_filename", "consultant", "called_at",
            "duration", "status", "error", "created_at", "processed_at",
            "evaluation", "transcript",
        ]


