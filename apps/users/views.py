from django.contrib import messages
from django.contrib.auth import login, logout
from django.shortcuts import redirect, render
from rest_framework.views import APIView

from .serializers import UserSerializer, UserLoginSerializer


class RegisterView(APIView):
    def get(self, request):
        return render(request, 'auth/registration.html')

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            if user:
                login(request, user)
                return redirect('select_subjects')
        return render(request, 'auth/registration.html', {'errors': serializer.errors})


class LoginView(APIView):
    def get(self, request):
        return render(request, 'auth/login.html')

    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        try:
            if serializer.is_valid():
                user = serializer.validated_data['user']
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid credentials.')
        except Exception as e:
            messages.error(request, str(e))

        return render(request, 'auth/login.html', {
            'errors': serializer.errors,
            'form_data': request.data  # To preserve form data on error
        })

class LogoutView(APIView):
    def get(self, request):
        if request.user.is_authenticated:
            # Perform any cleanup needed
            logout(request)
            # Add a success message if you want
            messages.success(request, 'You have been successfully logged out.')
        return redirect('home')