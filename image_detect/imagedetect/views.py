from django.shortcuts import render
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Create your views here.

class CountObjectsView(CreateAPIView):

    def post(self, request, *args, **kwargs):
        data = request.data
        model = data.get("image")

        img = cv2.imread('image1.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box, label, count = cv.detect_common_objects(img)
        # output = draw_bbox(img, box, label, count)

        return Response({"message": "success", "status": 200, "count": count}, status=status.HTTP_200_OK)
