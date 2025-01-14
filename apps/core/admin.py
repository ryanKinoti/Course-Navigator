from django.contrib import admin
from .models import Subject, Course, CourseRequirement


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'is_active')
    search_fields = ('name', 'code')
    list_filter = ('is_active',)


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('name', 'mean_grade','code')
    search_fields = ('name', 'mean_grade')
    list_filter = ('mean_grade',)
    # filter_horizontal = 'required_subjects'


@admin.register(CourseRequirement)
class CourseRequirementAdmin(admin.ModelAdmin):
    list_display = ('course', 'subject', 'minimum_grade', 'importance_weight')
    search_fields = ('course__name', 'subject__name', 'minimum_grade')
    list_filter = ('minimum_grade',)
