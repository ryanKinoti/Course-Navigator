from django.db import models


class Subject(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, unique=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


# apps/core/models.py
class Course(models.Model):
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField()
    mean_grade = models.CharField(max_length=2)  # e.g., 'B+', 'A-'
    university = models.CharField(max_length=200)
    course_url = models.URLField(max_length=500, blank=True, null=True)
    required_subjects = models.ManyToManyField(
        Subject,
        through='CourseRequirement'
    )

    def __str__(self):
        return f"{self.name} - {self.university}"


class CourseRequirement(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    minimum_grade = models.CharField(max_length=2)  # e.g., 'B+', 'A-'
    importance_weight = models.FloatField(default=1.0)

    class Meta:
        unique_together = ['course', 'subject']
