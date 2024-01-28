import cv2
import pandas as pd
import numpy as np
import os

from utils.iou import load_detections, compute_iou
from utils.tracking_results import draw_tracking_results, save_tracking_results

next_track_id = 1

def create_similarity_matrix(detections, tracks):
    n = len(detections)
    m = len(tracks)
    similarity_matrix = np.zeros((n, m))
    
    for i, detection in enumerate(detections):
        for j, track in enumerate(tracks):
            similarity_matrix[i][j] = compute_iou(detection, track['box'])
    return similarity_matrix

def associate_detections_to_tracks(similarity_matrix, sigma_iou):
    matches, unmatched_detections, unmatched_tracks = [], [], []

    if similarity_matrix.size == 0:
        unmatched_detections = list(range(similarity_matrix.shape[0]))
        unmatched_tracks = list(range(similarity_matrix.shape[1]))
        return matches, unmatched_detections, unmatched_tracks
    
    forbidden_idx = []

    for det_idx, row in enumerate(similarity_matrix):
        track_idx = row.argmax()
        max_iou = row[track_idx]
        while len(forbidden_idx) < len(row) and track_idx in forbidden_idx:
            row[track_idx] = -1
            track_idx = row.argmax()
            max_iou = row[track_idx]
        if max_iou > sigma_iou:
            matches.append((det_idx, track_idx))
            forbidden_idx.append(track_idx)
        else:
            unmatched_detections.append(det_idx)
    matched_tracks = [track_idx for _, track_idx in matches]
    unmatched_tracks = [i for i in range(similarity_matrix.shape[1]) if i not in matched_tracks]

    return matches, unmatched_detections, unmatched_tracks

def update_tracks(matches, unmatched_tracks, unmatched_detections, current_detections, current_confidences, previous_tracks, frame_number):
    global next_track_id
    for i, j in matches:
        box = current_detections[i]
        center = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
        previous_tracks[j]['box'] = box
        previous_tracks[j]['conf'] = current_confidences[i]
        previous_tracks[j]['frames'].append(frame_number)
        previous_tracks[j]['positions'].append(center)

    for i in unmatched_detections:
        new_track = {
            'id': next_track_id,
            'box': current_detections[i],
            'conf': current_confidences[i],
            'frames': [frame_number],
            'positions': [(int(current_detections[i][0] + current_detections[i][2] / 2), int(current_detections[i][1] + current_detections[i][3] / 2))]
        }
        previous_tracks.append(new_track)
        next_track_id += 1

    for i in reversed(unmatched_tracks):
        del previous_tracks[i]
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
sigma_iou = 0.2

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
        new_boxes = new_detections_df[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        new_confidences = new_detections_df['conf'].values
        if frame_number > 1:
            similarity_matrix = create_similarity_matrix(new_boxes, tracks)
            matches, unmatched_detections, unmatched_tracks = associate_detections_to_tracks(similarity_matrix, sigma_iou=0.5)
            tracks = update_tracks(matches, unmatched_tracks, unmatched_detections, new_boxes, new_confidences, tracks, frame_number)
        else:
            for i, row in new_detections_df.iterrows():
                initial_center = (int(row['bb_left'] + row['bb_width'] / 2), int(row['bb_top'] + row['bb_height'] / 2))
                tracks.append({
                    'id': row['id'],
                    'box': [row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']],
                    'conf': row['conf'],
                    'frames': [frame_number],
                    'positions': [initial_center]
                })
        frame_with_tracking = draw_tracking_results(frame, tracks)
        save_tracking_results(tracks, tracking_file, frame_number)
        cv2.imshow('Tracking', frame_with_tracking)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        out.write(frame_with_tracking)
        frame_number += 1
out.release()
cv2.destroyAllWindows()