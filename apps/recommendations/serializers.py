from rest_framework import serializers
from .models import StudentSubject, Recommendation, UserProfile
from apps.core.serializers import SubjectSerializer, CourseSerializer
from ..core.models import Subject


class StudentSubjectSerializer(serializers.ModelSerializer):
    subject_name = serializers.CharField(source='subject.name', read_only=True)

    class Meta:
        model = StudentSubject
        fields = ['id', 'subject', 'subject_name', 'grade', 'created_at']
        read_only_fields = ['created_at']


class GradeInputSerializer(serializers.Serializer):
    subject_id = serializers.IntegerField()
    grade = serializers.CharField()

    def validate_subject_id(self, value):
        try:
            Subject.objects.get(id=value)
            return value
        except Subject.DoesNotExist:
            raise serializers.ValidationError(f"Subject with id {value} does not exist")

    def validate_grade(self, value):
        valid_grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E']
        if value.upper() not in valid_grades:
            raise serializers.ValidationError("Invalid grade format")
        return value.upper()


class RecommendationSerializer(serializers.ModelSerializer):
    course = CourseSerializer(read_only=True)
    similarity_percentage = serializers.FloatField(read_only=True)
    confidence_percentage = serializers.FloatField(read_only=True)
    combined_percentage = serializers.FloatField(read_only=True)

    class Meta:
        model = Recommendation
        fields = [
            'id', 'course', 'similarity_score', 'confidence_score',
            'combined_score', 'created_at', 'similarity_percentage',
            'confidence_percentage', 'combined_percentage'
        ]
        read_only_fields = ['created_at']


class SubjectSelectionSerializer(serializers.ModelSerializer):
    subjects = SubjectSerializer(many=True, read_only=True)
    subject_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True
    )

    class Meta:
        model = UserProfile
        fields = ['id', 'subjects', 'subject_ids', 'mean_grade']
        read_only_fields = ['mean_grade']

    def validate_subject_ids(self, value):
        if len(value) > 8:
            raise serializers.ValidationError("You can only select up to 8 subjects")
        return value

    def update(self, instance, validated_data):
        subject_ids = validated_data.pop('subject_ids', [])
        instance.selected_subjects.set(subject_ids)
        return instance


class UserProfileSerializer(serializers.ModelSerializer):
    selected_subjects = SubjectSerializer(many=True, read_only=True)
    student_subjects = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = ['id', 'selected_subjects', 'mean_grade',
                  'last_recommendation_date', 'student_subjects']

    def get_student_subjects(self, obj):
        student_subjects = StudentSubject.objects.filter(user=obj.user)
        return StudentSubjectSerializer(student_subjects, many=True).data


class RecommendationRequestSerializer(serializers.Serializer):
    grades = GradeInputSerializer(many=True)

    def validate_grades(self, value):
        if not value:
            raise serializers.ValidationError("At least one grade is required")
        return value
