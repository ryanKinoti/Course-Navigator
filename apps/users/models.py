from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    email = models.EmailField(unique=True)
    exam_type = models.CharField(max_length=10, choices=[
        ('KCSE', 'KCSE'),
        ('IGCSE', 'IGCSE')
    ])

    def __str__(self):
        return self.email
