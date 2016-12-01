from django.contrib import admin
from .models import ImageModel
# Register your models here.

class ImageAdmin(admin.ModelAdmin):
	list_display = ('image',)


admin.site.register(ImageModel,ImageAdmin)
