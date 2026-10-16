from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class UserRole(models.TextChoices):
        CONSULTANT = "consultant", "консультант"
        SUPERVISOR = "supervisor", "Руководитель"

    role = models.CharField(
        max_length=20, choices=UserRole.choices, default=UserRole.CONSULTANT, db_index=True
    )

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    @property
    def is_supervisor(self) -> bool:
        """Руководитель и суперпользователь видят и обслуживают все звонки."""
        return self.is_superuser or self.role == self.UserRole.SUPERVISOR

    @property
    def display_name(self) -> str:
        return self.get_full_name() or self.username

    def deactivate_user(self):
        self.is_active = False
        self.save()

    def __str__(self):
        return self.display_name
