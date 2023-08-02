import random

from django.shortcuts import render
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Create your views here.

def convert_response(count):
    # Define the products/categories based on the provided shelf configuration
    products = {
        "PET 330 ML": "categ_1",
        "GLASS 270ML STILL SPARKLING": "categ_2",
        # Add more product-category mappings here as needed
    }

    # Create the shelfinfo list with dictionaries representing each shelf
    shelfinfo = {}
    for x in range(1, 5):
        shelf_name = "shelf_{}".format(x)
        shelfinfo[shelf_name] = [{
            "PET 330 ML": random.randint(0, count),
            "GLASS 270ML STILL ": random.randint(0, count),
            "GLASS 270ML SPARKLING": random.randint(0, count),
            "PET 600ML": random.randint(0, count),
            "GLASS 750ML": random.randint(0, count),

        }]

    # Construct the final response in the desired format
    output_response = {
        "shelfinfo": shelfinfo,
    }

    return output_response



class CountObjectsView(CreateAPIView):

    def post(self, request, *args, **kwargs):
        
        data = request.data
        model = data.get("image")
        with open("temp_image.jpg", "wb") as f:
             f.write(model.read())

        img = cv2.imread("temp_image.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box, label, count = cv.detect_common_objects(img)
        os.remove("temp_image.jpg")

        data = convert_response(count)

        return Response({"message": "success", "status": 200,"data":data}, status=status.HTTP_200_OK)
