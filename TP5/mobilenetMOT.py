import cv2
import pandas as pd
import numpy as np
import os

from scipy.optimize import linear_sum_assignment
from KallmanFilter import KalmanFilter
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from utils.iou import load_detections, compute_iou
from utils.tracking_results import draw_tracking_results, save_tracking_results

next_track_id = 1
mobilenet_model = None
preprocess = None

def init_model():
    global mobilenet_model, preprocess

    mobilenet_model = models.mobilenet_v2(pretrained=True)
    mobilenet_model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_predicted_box(kf, w, h):
    predicted_state = kf.x
    x_center, y_center = predicted_state[0, 0], predicted_state[1, 0]
    return [x_center - w / 2, y_center - h / 2, w, h]

def get_features(img, bbox):
    x, y, w, h = bbox
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    if crop_img.size == 0:
        return torch.zeros((1, 1000))
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_img)
    if (crop_img != None):
        input_tensor = preprocess(crop_img)
        input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = mobilenet_model(input_batch)
    return output

def features_similarity(iou, deep_features1, deep_features2):
    return iou * torch.cosine_similarity(deep_features1, deep_features2)

def create_similarity_matrix(det, track):
    similarity_matrix = np.zeros((len(det), len(track)))
    for i, detec in enumerate(det):
        deep_det = get_features(frame, detec)
        for j, prev_track in enumerate(track):
            predicted_box = get_predicted_box(prev_track['kf'], detec[2], detec[3])
            similarity_matrix[i][j] = features_similarity(compute_iou(detec, predicted_box), deep_det, prev_track['deep_features'])
    return similarity_matrix


def associate_detections_to_tracks(similarity_matrix, threshold):
    if similarity_matrix.size == 0:
        return [], list(range(similarity_matrix.shape[0])), list(range(similarity_matrix.shape[1]))
    
    cost_matrix = 1 - similarity_matrix
    det_indices, track_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_detections = list(range(similarity_matrix.shape[0]))
    unmatched_tracks = list(range(similarity_matrix.shape[1]))

    for det_idx, track_idx in zip(det_indices, track_indices):
        if similarity_matrix[det_idx, track_idx] >= threshold:
            matches.append((det_idx, track_idx))
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_idx)

    return matches, unmatched_detections, unmatched_tracks

def initialize_new_track(det, conf, frame_number):
    global next_track_id
    kf = KalmanFilter()
    kf.x = np.matrix([[det[0] + det[2] / 2], [det[1] + det[3] / 2], [0], [0]])
    deep_features = get_features(frame, det)
    new_track = {
        'id': next_track_id,
        'box': det,
        'conf': conf,
        'frames': [frame_number],
        'positions': [(int(det[0] + det[2] / 2), int(det[1] + det[3] / 2))],
        'kf': kf,
        'deep_features': deep_features
    }
    next_track_id += 1
    return new_track

def update_track(track, det, conf, frame_number):
    track['kf'].update([det[0] + det[2] / 2, det[1] + det[3] / 2])
    estimated_pos = track['kf'].x[:2]
    track['box'] = [estimated_pos[0, 0] - det[2] / 2, estimated_pos[1, 0] - det[3] / 2, det[2], det[3]]
    track['frames'].append(frame_number)
    track['positions'].append((int(estimated_pos[0, 0]), int(estimated_pos[1, 0])))
    track['conf'] = conf

def update_tracks(matches, unmatched_tracks, unmatched_detections, current_detections, current_confidences, previous_tracks, frame_number):

    for det_index, track_index in matches:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        update_track(previous_tracks[track_index], det, conf, frame_number)

    for det_index in unmatched_detections:
        det = current_detections[det_index]
        conf = current_confidences[det_index]
        new_track = initialize_new_track(det, conf, frame_number)
        previous_tracks.append(new_track)

    previous_tracks[:] = [track for index, track in enumerate(previous_tracks) if index not in unmatched_tracks]

    return previous_tracks

det_file_path = '../data/det/det.txt'
detections_df = load_detections(det_file_path)
images_path = '../data/img1/' 
fps = 30
frame_size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
delay = int(1000/fps)
tracks = []
frame_number = 1
sigma_iou = 0.5
CONF_THRESH = 40.0

init_model()

tracking_file = 'ADL-Rundle-6.txt'
if os.path.isfile(tracking_file):
    open(tracking_file, 'w').close()
else:
    open(tracking_file, 'x')

for filename in sorted(os.listdir(images_path)):
    if filename.endswith(".jpg"):
        frame_path = os.path.join(images_path, filename)
        frame = cv2.imread(frame_path)
        new_detections_df = detections_df[detections_df['frame'] == frame_number]
        new_detections_df = new_detections_df[new_detections_df['conf'] >= CONF_THRESH]
        new_boxes = new_detections_df[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        new_confidences = new_detections_df['conf'].values
        for track in tracks:
            track['kf'].predict()
        if frame_number > 1:
            similarity_matrix = create_similarity_matrix(new_boxes, tracks)
            matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(similarity_matrix, threshold=0.5)
            tracks = update_tracks(matches, unmatched_tracks, unmatched_detections, new_boxes, new_confidences, tracks, frame_number)
        else:
            for i, det in enumerate(new_boxes):
                tracks.append(initialize_new_track(det, new_confidences[i], frame_number))
        frame_with_tracking = draw_tracking_results(frame, tracks)
        save_tracking_results(tracks, tracking_file, frame_number)
        cv2.imshow('Tracking', frame_with_tracking)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        out.write(frame_with_tracking)
        frame_number += 1
out.release()
cv2.destroyAllWindows()