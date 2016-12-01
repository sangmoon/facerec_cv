from __future__ import unicode_literals

from django.db import models

# Create your models here.

class ImageModel(models.Model):
	image = models.ImageField(upload_to = 'cv/upload/', default= 'upload_img/None/no-img.jpg')
	