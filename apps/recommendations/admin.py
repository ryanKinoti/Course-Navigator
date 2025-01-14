from django.contrib import admin
from .models import StudentSubject, Recommendation, UserProfile

@admin.register(StudentSubject)
class StudentSubjectAdmin(admin.ModelAdmin):
    list_display = ['user', 'subject', 'grade', 'created_at']
    list_filter = ['grade', 'created_at']
    search_fields = ['user__email', 'subject__name']

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['user', 'course', 'similarity_score', 'confidence_score', 'combined_score', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__email', 'course__name']

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'mean_grade', 'last_recommendation_date']
    list_filter = ['last_recommendation_date']
    search_fields = ['user__email']