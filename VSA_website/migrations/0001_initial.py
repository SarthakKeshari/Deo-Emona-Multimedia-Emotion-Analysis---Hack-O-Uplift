# Generated by Django 3.2.6 on 2021-08-28 22:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Doubt',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('email', models.CharField(max_length=100)),
                ('subject', models.CharField(max_length=150)),
                ('doubt_desc', models.TextField()),
                ('date', models.DateField()),
            ],
        ),
        migrations.CreateModel(
            name ='Video_Upload',
            fields =[
                ('video_name', models.IntegerField()),
                ('video_file', models.FileField()),
            ],
        ),
    ]