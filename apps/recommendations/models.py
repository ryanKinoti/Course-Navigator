from django.db import models
from django.conf import settings
from apps.core.models import Subject, Course


class StudentSubject(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    grade = models.CharField(max_length=2)  # e.g., 'A', 'B+'
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['user', 'subject']


class Recommendation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    similarity_score = models.FloatField()
    confidence_score = models.FloatField()
    combined_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-combined_score']

    @property
    def similarity_percentage(self):
        """Convert similarity score to percentage"""
        return round(self.similarity_score * 100, 1)

    @property
    def confidence_percentage(self):
        """Convert confidence score to percentage"""
        return round(self.confidence_score * 100, 1)

    @property
    def combined_percentage(self):
        """Convert combined score to percentage"""
        return round(self.combined_score * 100, 1)


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    selected_subjects = models.ManyToManyField(Subject)
    mean_grade = models.FloatField(null=True, blank=True)
    last_recommendation_date = models.DateTimeField(null=True, blank=True)

    def calculate_mean_grade(self):
        grades = StudentSubject.objects.filter(user=self.user)
        if not grades.exists():
            return None

        grade_values = {
            'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0,
            'D-': 0.7, 'E': 0.3
        }

        inverse_grade_values = {v: k for k, v in grade_values.items()}  # Reverse mapping

        total = sum(grade_values.get(grade.grade, 0) for grade in grades)
        mean_value = total / grades.count()

        # Find the closest grade value, favoring higher grades in a tie
        closest_grade = min(inverse_grade_values.keys(), key=lambda x: (abs(x - mean_value), -x))

        # Return the grade letter
        return inverse_grade_values[closest_grade]
