# Generated by Django 3.2.6 on 2021-09-04 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('VSA_website', '0005_doubt_question'),
    ]

    operations = [
        migrations.CreateModel(
            name='Config',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('api_key', models.CharField(max_length=200)),
                ('identifier', models.CharField(max_length=200)),
            ],
        ),
    ]
