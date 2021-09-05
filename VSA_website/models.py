from django.db import models

# makemigrations - create changes and store in a file
# migrate - apply the pending changes created by makemigrations

# Create your models here.
class Doubt(models.Model):
    name = models.CharField(max_length=100)
    # email = models.EmailField(max_length=100)
    email = models.CharField(max_length=100)
    subject = models.CharField(max_length=150)
    doubt_desc = models.TextField()
    date = models.DateField()
    time = models.TimeField()
    question = models.TextField()
    answer = models.TextField()

    def __str__(self):
        return "Name: "+self.name+" | Subject: "+self.subject
    
class Video_Upload(models.Model):
    video_name = models.IntegerField(default=0)
    video_file = models.FileField(upload_to='videos/', null=True, verbose_name="")

    def __str__(self):
        return "Name: "+str(self.video_name)

class Config(models.Model):
    api_key = models.CharField(max_length=200)
    identifier = models.CharField(max_length=200)
    choices = [("email","Email"),("phone_number_sms","Phone")]