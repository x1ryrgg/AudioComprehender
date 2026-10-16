from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from call_recognition.models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (("Роль", {"fields": ("role",)}),)
    add_fieldsets = UserAdmin.add_fieldsets + (("Роль", {"fields": ("first_name", "last_name", "role")}),)
    list_display = ("username", "first_name", "last_name", "role", "is_active")
    list_filter = UserAdmin.list_filter + ("role",)