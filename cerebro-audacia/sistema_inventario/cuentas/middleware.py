from datetime import timedelta

from django.utils import timezone


class SesionAudaciaAdminMiddleware:
    """Renueva la sesión del superusuario AudaciaAdmin mientras siga en uso."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        user = request.user
        if user.is_authenticated and user.is_superuser and user.get_username() == "AudaciaAdmin":
            ahora = timezone.now()
            if request.session.get_expiry_date() < ahora + timedelta(days=370):
                request.session.set_expiry(ahora + timedelta(days=400))
        return response
