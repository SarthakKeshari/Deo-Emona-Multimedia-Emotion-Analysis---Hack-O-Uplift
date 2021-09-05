from VSA_website.models import Video_Upload
from django import forms
from .models import Video_Upload

class VideoForm(forms.ModelForm):
    class Meta:
        model= Video_Upload
        fields= ["video_name", "video_file"]
