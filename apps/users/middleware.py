# apps/users/middleware.py
from apps.recommendations.models import UserProfile

class UserProfileMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            UserProfile.objects.get_or_create(user=request.user)
        return self.get_response(request)