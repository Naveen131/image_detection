import base64
import random

from botocore.exceptions import ClientError
from django.shortcuts import render
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
import os

import boto3
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont


def process_custom_labels(response, num_shelves):
    shelf_labels = {}  # Dictionary to store labels inside each shelf
    for customLabel in response['CustomLabels']:
        label_name = customLabel['Name']
        # Skip 'Shelf' label
        if label_name == 'Shelf':
            continue

        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = box['Left']
            top = box['Top']

            # Determine which shelf this label belongs to
            shelf_index = int(top * num_shelves) + 1

            # Create shelf entry if it doesn't exist
            if shelf_index not in shelf_labels:
                shelf_labels[shelf_index] = {}

            # Count label occurrences within each shelf
            if label_name in shelf_labels[shelf_index]:
                shelf_labels[shelf_index][label_name] += 1
            else:
                shelf_labels[shelf_index][label_name] = 1

        formatted_data ={}
        for shelf_number, labels in shelf_labels.items():
            # Sort labels by count in descending order
            sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
            formatted_data[f"shelf_{shelf_number}"] = {label: count for label, count in sorted_labels}

    return formatted_data

def find_number_of_shelves(response):
    num_shelves = 0
    for customLabel in response['CustomLabels']:
        label_name = customLabel['Name']
        if label_name == 'Shelf' and 'Geometry' in customLabel:
            num_shelves += 1
    return num_shelves

def upload_file_to_s3(base64_data, object_name):
    # import pdb;pdb.set_trace()
    image_data = base64.b64decode(base64_data)

    s3_client = boto3.client('s3', aws_access_key_id='AKIA5MU5LBBEOBOA2HCQ', aws_secret_access_key='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',region_name='ap-south-1' )
    try:
        response = s3_client.put_object(Body=image_data, Bucket='analyse-invoice', Key=object_name, ACL='public-read')
        s3_url = f"https://analyse-invoice.s3.ap-south-1.amazonaws.com/{object_name}"
        return s3_url
    except ClientError as e:
        print("Error:", e)


def draw_bounding_boxes(bucket, photo, response):
    s3 = boto3.client('s3')
    img = s3.get_object(Bucket=bucket, Key=photo)
    image_bytes = img['Body'].read()
    image = Image.open(io.BytesIO(image_bytes))

    draw = ImageDraw.Draw(image)

    for label in response['CustomLabels']:
        if label['Geometry']:
            bbox = label['Geometry']['BoundingBox']
            h, w = image.size
            x = int(bbox['Left'] * w)
            y = int(bbox['Top'] * h)
            width = int(bbox['Width'] * w)
            height = int(bbox['Height'] * h)
            draw.rectangle([x, y, x + width, y + height], outline=(0, 255, 0), width=2)

    return image

def convert_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_encoded_image

def display_image(bucket, photo, response):
    # Load image from S3 bucket
    s3_connection = boto3.resource('s3')

    s3_object = s3_connection.Object(bucket, photo)
    s3_response = s3_object.get()

    stream = io.BytesIO(s3_response['Body'].read())
    image = Image.open(stream)

    # Ready image to draw bounding boxes on it.
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    # calculate and display bounding boxes for each detected custom label
    print('Detected custom labels for ' + photo)
    for customLabel in response['CustomLabels']:
        print('Label ' + str(customLabel['Name']))
        print('Confidence ' + str(customLabel['Confidence']))
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
            draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)

            print('Left: ' + '{0:.0f}'.format(left))
            print('Top: ' + '{0:.0f}'.format(top))
            print('Label Width: ' + "{0:.0f}".format(width))
            print('Label Height: ' + "{0:.0f}".format(height))

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top))
            draw.line(points, fill='#00d400', width=5)

    image.show()


def show_custom_labels(model, bucket, photo, min_confidence):

        client = boto3.client('rekognition', aws_access_key_id='AKIA5MU5LBBEOBOA2HCQ', aws_secret_access_key='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',region_name='ap-south-1' )

        # Call DetectCustomLabels
        response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
                                               MinConfidence=min_confidence,
                                               ProjectVersionArn=model)

        # import pdb;pdb.set_trace()
        num_shelves = find_number_of_shelves(response)

        # Process custom labels and count occurrences within shelves
        shelf_labels  = process_custom_labels(response, num_shelves)
        # # For object detection use case, uncomment below code to display image.
        # image = display_image(bucket,photo,response)
        # shelf_labels['image'] = base64.b64encode(image).decode("utf-8")

        image_with_bboxes = draw_bounding_boxes(bucket, photo, response)

        # Convert the image to base64
        base64_encoded_image = convert_to_base64(image_with_bboxes)

        shelf_labels['image'] = base64_encoded_image

        return shelf_labels


def start_model(project_arn, model_arn, version_name, min_inference_units):
    client = boto3.client('rekognition')

    try:
        # Start the model
        print('Starting model: ' + model_arn)
        response = client.start_project_version(ProjectVersionArn=model_arn, MinInferenceUnits=min_inference_units)
        # Wait for the model to be in the running state
        project_version_running_waiter = client.get_waiter('project_version_running')
        project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])

        # Get the running status
        describe_response = client.describe_project_versions(ProjectArn=project_arn,
                                                             VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage'])
    except Exception as e:
        print(e)

    print('Done...')


# Create your views here.

# def convert_response(count):
#     # Define the products/categories based on the provided shelf configuration
#     products = {
#         "PET 330 ML": "categ_1",
#         "GLASS 270ML STILL SPARKLING": "categ_2",
#         # Add more product-category mappings here as needed
#     }
#
#     # Create the shelfinfo list with dictionaries representing each shelf
#     shelfinfo = {}
#     for x in range(1, 5):
#         shelf_name = "shelf_{}".format(x)
#
#         shelfinfo[shelf_name] = {
#             "PET 330 ML": random.randint(0, count),
#             "GLASS 270ML STILL ": random.randint(0, count),
#             "GLASS 270ML SPARKLING": random.randint(0, count),
#             "PET 600ML": random.randint(0, count),
#             "GLASS 750ML": random.randint(0, count)
#         }
#
#     # Construct the final response in the desired format
#     output_response = {
#         "shelfinfo": shelfinfo,
#     }
#
#     return output_response



class CountObjectsView(CreateAPIView):

    def post(self, request, *args, **kwargs):
        
        data = request.data
        img = data.get("image")



        file_content = img.read()

        # Encode the file content as base64
        base64_data = base64.b64encode(file_content).decode("utf-8")

        file_content = upload_file_to_s3(base64_data, img.name)

        # project_arn = 'arn:aws:rekognition:ap-south-1:920526719048:project/shelf_detection/1692858605176'
        # model_arn = 'arn:aws:rekognition:ap-south-1:920526719048:project/shelf_detection/version/shelf_detection.2023-08-24T19.38.25/1692886099867'
        # min_inference_units = 1
        # version_name = 'shelf_detection.2023-08-24T19.38.25'
        # start_model(project_arn, model_arn, version_name, min_inference_units)

        bucket = 'analyse-invoice'
        photo = img.name
        model = 'arn:aws:rekognition:ap-south-1:920526719048:project/shelf_detection/version/shelf_detection.2023-08-26T16.00.15/1693045814311'
        min_confidence = 50


        label_count = show_custom_labels(model, bucket, photo, min_confidence)
        print("Custom labels detected: " + str(label_count))
        # with open("temp_image.jpg", "wb") as f:
        #      f.write(model.read())
        #
        # img = cv2.imread("temp_image.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # box, label, count = cv.detect_common_objects(img)
        # os.remove("temp_image.jpg")
        #
        # data = convert_response(len(count))
        #

        # display_image(bucket, photo, response)


        return Response({"message": "success", "status": 200,"data":label_count}, status=status.HTTP_200_OK)
