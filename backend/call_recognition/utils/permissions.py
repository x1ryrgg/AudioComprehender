from rest_framework.permissions import BasePermission


class CallPermission(BasePermission):
    """Любой авторизованный пользователь может загружать звонки и смотреть доступные ему
    (консультант видит свои, руководитель все). Менять и удалять звонки может только руководитель."""

    restricted_actions = {"update", "partial_update", "destroy"}

    def has_permission(self, request, view):
        user = request.user
        if not (user and user.is_authenticated):
            return False
        if getattr(view, "action", None) in self.restricted_actions:
            return user.is_supervisor
        return True


class IsSupervisor(BasePermission):
    """Доступ только руководителю (или суперпользователю)."""

    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and user.is_supervisor)