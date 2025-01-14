from django.urls import path
from .views import (
    SubjectSelectionView,
    DashboardView,
    RecommendationView
)

urlpatterns = [
    path('select-subjects/', SubjectSelectionView.as_view(), name='select_subjects'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('results/', RecommendationView.as_view(), name='recommendations'),
]
