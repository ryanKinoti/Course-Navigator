from django.urls import path
from .views import HomeView, CourseListView, CourseDetailView

urlpatterns = [
    path('', HomeView.as_view(), name='home'),  # Landing page
    path('courses/', CourseListView.as_view(), name='course-list'),  # List all available courses
    path('courses/<int:pk>/', CourseDetailView.as_view(), name='course-detail'),  # Course details
]