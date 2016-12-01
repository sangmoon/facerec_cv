from django import forms

class UploadFileForm(forms.Form):
    """ Image upload form."""
    # title = forms.CharField(max_length = 50)
    image = forms.ImageField()
