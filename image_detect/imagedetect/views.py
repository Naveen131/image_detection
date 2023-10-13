import base64

from botocore.exceptions import ClientError

from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
import os

import boto3
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont

from image_detect import settings


def determine_shelf(label_top, shelf_labels, num_shelves):
    # Calculate the height of each shelf segment based on the number of shelves
    shelf_height = 1.0 / num_shelves

    # Iterate through each potential shelf
    for shelf_index in range(1, num_shelves + 1):
        # Calculate the top and bottom boundaries of the shelf segment
        shelf_top = (shelf_index - 1) * shelf_height
        shelf_bottom = shelf_index * shelf_height

        # Check if the label_top falls within the boundaries of the current shelf
        if shelf_top <= label_top <= shelf_bottom:
            return shelf_index

    # If the label_top doesn't fall within any shelf, return None
    return None


def process_custom_labels(response, num_shelves):


    print("shelfes are:", num_shelves)
    formatted_data = {}
    print("response", response)
    shelf_labels = {}  # Dictionary to store labels inside each shelf
    # import pdb;pdb.set_trace()
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
            shelf_index = round(top * num_shelves)

            print("shelf_index", shelf_index)
            # Create shelf entry if it doesn't exist
            if shelf_index not in shelf_labels:
                shelf_labels[shelf_index] = {}

            # Count label occurrences within each shelf
            if label_name in shelf_labels[shelf_index]:
                shelf_labels[shelf_index][label_name] += 1
            else:
                shelf_labels[shelf_index][label_name] = 1

        for shelf_number, labels in shelf_labels.items():
            # Sort labels by count in descending order
            sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
            formatted_data[f"shelf_{shelf_number}"] = {label: count for label, count in sorted_labels}

            # Include the last shelf
        # last_shelf_labels = shelf_labels[num_shelves-1]
        # sorted_labels = sorted(last_shelf_labels.items(), key=lambda x: x[1], reverse=True)
        # formatted_data[f"shelf_{num_shelves}"] = {label: count for label, count in sorted_labels}

    return formatted_data

def find_number_of_shelves(response):
    num_shelves = 0
    for customLabel in response['CustomLabels']:
        label_name = customLabel['Name']
        if label_name == 'Shelf' and 'Geometry' in customLabel:
            num_shelves += 1
    return num_shelves

def upload_file_to_s3(base64_data, object_name):
    image_data = base64.b64decode(base64_data)

    s3_client = boto3.client('s3', aws_access_key_id=settings.aws_access_key_id,
                             aws_secret_access_key=settings.aws_secret_access_key,
                             region_name=settings.region_name )
    try:
        response = s3_client.put_object(Body=image_data, Bucket=settings.bucket, Key=object_name)
        s3_url = f"https://berain-detection.s3.ap-south-1.amazonaws.com/{object_name}"
        return s3_url
    except ClientError as e:
        print("Error:", e)


def draw_bounding_boxes(bucket, photo, response):
    s3 = boto3.client('s3', aws_access_key_id=settings.aws_access_key_id,
                      aws_secret_access_key=settings.aws_secret_access_key,
                      region_name=settings.region_name )
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
    s3_connection = boto3.resource('s3', aws_access_key_id=settings.aws_access_key_id,
                                   aws_secret_access_key=settings.aws_secret_access_key,
                                   region_name=settings.region_name )

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
        # print('Label ' + str(customLabel['Name']))
        # print('Confidence ' + str(customLabel['Confidence']))
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.load_default()
            draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)

            # print('Left: ' + '{0:.0f}'.format(left))
            # print('Top: ' + '{0:.0f}'.format(top))
            # print('Label Width: ' + "{0:.0f}".format(width))
            # print('Label Height: ' + "{0:.0f}".format(height))

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top))
            draw.line(points, fill='#00d400', width=5)

    # image.show()

    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    base64_image = base64.b64encode(buffered.getvalue()).decode()

    return base64_image


def count_labels_inside_shelves(response):
    shelf_counts = {}
    shelf_counter = 1  # Initialize a counter for naming the shelves

    for label in response['CustomLabels']:
        if label['Name'].startswith('Shelf') and 'Geometry' in label:
            shelf_name = f'shelf_{shelf_counter}'  # Name the shelf
            shelf_counter += 1  # Increment the counter

            shelf_box = label['Geometry']['BoundingBox']
            shelf_left = shelf_box['Left']
            shelf_top = shelf_box['Top']
            shelf_width = shelf_box['Width']
            shelf_height = shelf_box['Height']

            labels_inside_shelf = {}

            for other_label in response['CustomLabels']:

                other_label_name = other_label['Name']
                if 'Geometry' in other_label:
                    other_label_box = other_label['Geometry']['BoundingBox']
                    other_label_left = other_label_box['Left']
                    other_label_top = other_label_box['Top']

                    # Check if the label is inside the shelf
                    if (shelf_left <= other_label_left <= (shelf_left + shelf_width) and
                        shelf_top <= other_label_top <= (shelf_top + shelf_height)):
                        if other_label_name in labels_inside_shelf:
                            labels_inside_shelf[other_label_name] += 1
                        else:
                            labels_inside_shelf[other_label_name] = 1

            shelf_counts[shelf_name] = labels_inside_shelf


            output_data = {}
            for shelf_name, shelf_data in shelf_counts.items():
                # Remove 'Shelf' from the shelf's dictionary
                cleaned_shelf_data = {key: value for key, value in shelf_data.items() if key != "Shelf"}
                output_data[shelf_name] = cleaned_shelf_data

            new_response = {
                f"shelf_{6 - int(key.split('_')[1])}": value
                for key, value in output_data.items()
            }

    return new_response
        

def show_custom_labels(model, bucket, photo, min_confidence, base64_data):
        # import pdb;pdb.set_trace()

        client = boto3.client('rekognition', aws_access_key_id=settings.aws_access_key_id,
                      aws_secret_access_key=settings.aws_secret_access_key,
                      region_name=settings.region_name )



        image_bytes = base64.b64decode(base64_data)

        # Make the DetectLabels API request
        response = client.detect_custom_labels(
            Image={
                "Bytes": image_bytes
            },
            MinConfidence=min_confidence,
            ProjectVersionArn=model

        )

        # import pdb;pdb.set_trace()
        print(response)
        num_shelves = find_number_of_shelves(response)

        # Process custom labels and count occurrences within shelves
        shelf_labels  = count_labels_inside_shelves(response)
        # # For object detection use case, uncomment below code to display image.
        # display_image(bucket,photo,response)
        shelf_labels['image'] = display_image(bucket,photo,response)

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


class CountObjectsView(CreateAPIView):

    def post(self, request, *args, **kwargs):
        data = request.data
        img = data.get("image")

        import datetime
        photo = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".jpeg"

        print("==================", img.content_type)

        # import pdb;pdb.set_trace()

        image_bytes = img.read()

        image_format = self.get_image_format(img.content_type)

        if not image_format:
            return Response({"message": "Invalid image format", "status": 400}, status=status.HTTP_400_BAD_REQUEST)

        image_buffer = io.BytesIO(image_bytes)

        # Open the image using PIL (Pillow) based on the detected format
        img = Image.open(image_buffer)

        # Resize the image if necessary
        img = self.resize_image(img)

        # Convert the image to JPEG format (if it's not already)
        if image_format != 'jpeg':
            img = img.convert('RGB')

        # Encode the image as base64
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        base64_data = base64.b64encode(buffered.getvalue()).decode()

        bucket = 'berain-detection'
        # photo = img.name.replace(".png", ".jpeg")

        model = 'arn:aws:rekognition:ap-south-1:864221354765:project/bottle/version/bottle.2023-10-12T13.21.10/1697097066154'
        min_confidence = 50
        upload_file_to_s3(base64_data, photo)
        # Call the show_custom_labels function to get the label count
        label_count = show_custom_labels(model, bucket, photo, min_confidence, base64_data)

        # Construct the response
        # response_data = {
        #     "message": "success",
        #     "status": 200,
        #     "data": {
        #         "label_count": label_count
        #     }
        # }
        # reversed_response = {
        #     "message": "success",
        #     "status": 200,
        #     "data": {}
        # }


        return Response({"message": "success", "status": 200,"data":label_count}, status=status.HTTP_200_OK)

    @staticmethod
    def get_image_format(content_type):
        # Determine the image format based on the content type
        if content_type == 'image/jpeg':
            return 'jpeg'
        elif content_type == 'image/png':
            return 'png'
        else:
            return None

    @staticmethod
    def resize_image(img):
        # Resize the image if needed (you can adjust the dimensions as required)
        max_width, max_height = 4096, 4096
        width, height = img.size
        if width > max_width or height > max_height:
            img.thumbnail((max_width, max_height), Image.LANCZOS)
        return img