from django.shortcuts import render
from django.http import HttpResponse, HttpResponseForbidden
from .forms import UploadFileForm
from .models import ImageModel
import cStringIO
import pickle
import cv2
import base64
from PIL import Image
from StringIO import StringIO
import numpy as np
from urllib2 import urlopen
# Create your views here.
def home(request):
    return render(request, 'home.html')

def preprocess(img1):
    from facerec.preprocessing import detect, draw_rects, crop_img
    cascade_fn = "/home/smant/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"

    cascade = cv2.CascadeClassifier(cascade_fn)

#    sbuf = StringIO()
#   sbuf.write(base64.b64decode(img1,"-_"))
#   Pimg=Image.open(sbuf)
    img = img1    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detect(gray, cascade)
    vis = crop_img(gray, rects) #remove background  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #contrast
    vis = clahe.apply(vis)
    vis = cv2.resize(vis, (100, 100)) #resize
    return vis, vis


    

def process(img):
    """ 0 female, 1 male"""
    import sys, os
    #sys.path.append("./facerec-master/py")
    from facerec.serialization import load_model
    from facerec.validation import MyValidation as MV
    model = load_model("/home/smant/snu/django_page/mysite/cv/myModel.pkl")

    cv = MV(model)
    gray,sangmoon=preprocess(img)
    result = cv.determine(gray)
    result_str=""

    if result==0:
        result_str="Female!!"
    else:
        result_str="Male!!"
    return result_str, sangmoon

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

def result(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            m = ImageModel()
            m.model_pic = form.cleaned_data['image']
            m.save()
            mStream = cStringIO.StringIO(m.model_pic.read())
            data_uri = 'data:image/jpg;base64,'
            
            data_uri += mStream.getvalue().encode('base64').replace('\n','')
            try:
                result_str, cv_img = process(create_opencv_image_from_stringio(mStream, 1))
            except:
                return HttpResponse("can't find face! please another image!")
            
            ret, buf = cv2.imencode( '.jpg', cv_img )
            
            b64 = cStringIO.StringIO(buf)
            cv_img_uri = 'data:image/jpg;base64,'
            cv_img_uri += b64.getvalue().encode('base64').replace('\n', '')
            
            return render(request, "result.html",{"cv_img_src":cv_img_uri, "img_src":data_uri, "result_str":result_str, },)

    return HttpResponseForbidden("allowed only POST")

