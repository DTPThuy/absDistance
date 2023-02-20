from __future__ import absolute_import, division, print_function

import torch
import cv2 as cv

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from util import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
import numpy as np
from PIL import Image
import copy

import time


#Return pandas DataFrame contains (x_min, y_min) and (x_max, y_max) and classes of object in frame
def model_yolov5(frame, model):
  # Model
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

  # Inference
  results = model(frame)
  results.print()
  return results.pandas().xyxy[0]


#Return an numpy array represents for depth map
def depth_predict(image_path, encoder, depth_decoder):

  original_width, original_height = image_path.shape[1], image_path.shape[0]

  # PREDICTING ON EACH IMAGE IN TURN
  with torch.no_grad():

      # Load image and preprocess
      image_path = cv.resize(
          image_path, (feed_width, feed_height), cv.INTER_LANCZOS4)
      image_path = transforms.ToTensor()(image_path).unsqueeze(0)

      # PREDICTION, encoder and decoder
      image_path = image_path.to(device)
      features = encoder(image_path)
      outputs = depth_decoder(features)

      disp = outputs[("disp", 0)]
      disp_resized = torch.nn.functional.interpolate(
          disp, (original_height, original_width), mode="bilinear", align_corners=False)

      scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
      metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

      metric_depth = resize_depth_map(
          metric_depth, original_width, original_height)

      return metric_depth


def resize_depth_map(metric_depth, original_width, original_height):
  metric_depth = torch.from_numpy(metric_depth)
  metric_depth_resized = torch.nn.functional.interpolate(metric_depth,
                                                         (original_height, original_width), mode="bilinear", align_corners=False)

  # Saving colormapped depth image
  metric_depth_resized_np = metric_depth_resized.squeeze().cpu().numpy()
  return metric_depth_resized_np


#Calculate relative distance of objects in the image from depth map and bounder box
#depth_map : nparray
#data: Dataframe obtains from yolov5
#return dataframe that contain collumn "rev_distance"
def calculate_rev(depth_map, data):
  rev_dis = []
  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])
    y_max = int(row[1]['ymax'])

    rev = 0
    num = (y_max - y_min) * (x_max - x_min)
    for i in range(y_min, y_max):
      for j in range(x_min, x_max):
        rev += depth_map[i, j]
    rev /= num
    rev_dis.append(rev)

  data['rev_distance'] = rev_dis
  return data


def calculate_abs(depth_map, data):
  abs_dis = []
  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])  # Wight
    y_max = int(row[1]['ymax'])  # Hight
    
    # Tính median của tất cả các giá trị depth trong bounding box
    depth_box = depth_map[y_min:y_max, x_min:x_max]
    median_depth = np.median(depth_box)

    # Tính khoảng cách tuyệt đối
    absolute_distance = ((-0.00056 * median_depth ** 2 +
                         0.146 * median_depth + 1.02) * 0.5) * 100

    abs_dis.append(absolute_distance)

  data['abs_distance'] = abs_dis
  return data


def calculate_abs2(depth_map, data, camera_height):
  abs_dis = []
  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])  # Wight
    y_max = int(row[1]['ymax'])  # Hight
    
    # Tính median của tất cả các giá trị depth trong bounding box
    depth_box = depth_map[y_min:y_max, x_min:x_max]
    median_depth = np.median(depth_box)
    
    # Fit a linear regression model to obtain the coefficients c0, c1, and c2
    X = np.array([[median_depth**2], [median_depth], [1]])
    Y = np.array([camera_height]).reshape((1, 1))
    
    model = LinearRegression().fit(X, Y)
    c0, c1, c2 = model.intercept_[0], model.coef_[1], model.coef_[2]

    # Calculate the absolute distance using equation (1)
    absolute_distance = (c0 + c1 * median_depth +
                         c2 * median_depth**2) * camera_height
    
    abs_dis.append(absolute_distance)

  data['abs_distance'] = abs_dis
  return data

#Drawing label and distance on frame
#frame: image
#data: dataFrame contain relative distance
def drawing_output(frame, model, encoder, depth_decoder):
  frame_temp = copy.copy(frame)
  y = model_yolov5(frame_temp, model)
  map = depth_predict(frame, encoder, depth_decoder)
  data = calculate_abs(map, y)

  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])
    y_max = int(row[1]['ymax'])

    name_label = row[1]['name']
    rev = row[1]['abs_distance']
    
    # x = rev
    # h = 0.5
    # c0, c1, c2 = 1.02, 0.146, -0.00056
    # abs_dis = ((c0 + c1*x + c2*x**2) * h ) * 100
    
    
    str_output = name_label + ": " + str(int(rev)) + "cm"
    # # TEST

    # # Example input: camera height and bounding box
    # camera_height = 1.5
    # bounding_box = [x_min, y_min, x_max, y_max]

    # # Call the function to calculate the absolute distance (ABS) from the camera
    # absolute_distance = calculate_abs_distance(
    #     map, camera_height, bounding_box)
    # cv.putText(frame, "abs {:.2f} cm".format(
    #     abs_dis), (x_min, y_max), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.rectangle(frame, (x_min, y_min),
                 (x_max, y_max),
                 (0, 0, 255), 2, 8)
    cv.putText(frame, str_output, (x_min, y_min),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv.LINE_AA)
  return frame


def calculate_abs_distance(depth_image, camera_height, bounding_box):
    # Extract the estimated distances of all the pixels inside the bounding box
    print("DepthMap nay de xem", depth_image)
    distances = depth_image[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
    print("bounding_box[0]", bounding_box[0])
    print("bounding_box[2]", bounding_box[2])
    print("bounding_box[1]", bounding_box[1])
    print("bounding_box[3]", bounding_box[3])
    # Compute the relative distance (REV) of the object
    relative_distance = np.median(distances)
    print("relative_distance", relative_distance)
    # Define the input X and the output Y for the curve fitting
    X = np.array([[relative_distance**2], [relative_distance], [1]])
    Y = np.array([camera_height])

    # Fit a linear regression model to obtain the coefficients c0, c1, and c2
    model = LinearRegression().fit(X, Y)
    c0, c1, c2 = model.coef_[0], model.coef_[1], model.coef_[2]

    # Calculate the absolute distance (ABS) using equation (1)
    absolute_distance = (c0 + c1 * relative_distance +
                         c2 * relative_distance**2) * camera_height

    return absolute_distance


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_name = "mono+stereo_640x192"

    #use GPU
    if torch.cuda.is_available():
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")

    #
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Open the camera
    cap = cv.VideoCapture(0)

    # Continuously capture images from the camera
    while True:

      # Read a depth frame from the camera
      ret, frame = cap.read()
      if ret == False:
        break

      frame = drawing_output(frame, model, encoder, depth_decoder)

      cv.imshow("Object Detection", frame)

      # Break the loop if the 'q' key is pressed
      if cv.waitKey(1) & 0xFF == ord('q'):
          break

    # Release the camera and close the window
    cap.release()
    cv.destroyAllWindows()
