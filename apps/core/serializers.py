from rest_framework import serializers
from .models import Subject, Course, CourseRequirement


class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = ['id', 'name', 'code', 'is_active']


class CourseRequirementSerializer(serializers.ModelSerializer):
    subject_name = serializers.CharField(source='subject.name', read_only=True)

    class Meta:
        model = CourseRequirement
        fields = ['subject', 'subject_name', 'minimum_grade', 'importance_weight']


class CourseSerializer(serializers.ModelSerializer):
    requirements = CourseRequirementSerializer(
        source='courserequirement_set',
        many=True,
        read_only=True
    )

    class Meta:
        model = Course
        fields = ['id', 'name', 'description', 'mean_grade', 'requirements']
