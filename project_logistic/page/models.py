from django.db import models

# Create your models here.

class Example(models.Model):
    headling = models.CharField(max_length=255)
    in_completed = models.CharField(max_length=200)
    name = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.headling