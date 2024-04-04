import os
import cv2
import PySimpleGUI as sg
import numpy as np

def process_image(sample_path, real_path, progress_bar):
    sample = cv2.imread(sample_path, 0)
    best_score = 0
    best_match = None

    total_files = len(os.listdir(real_path))
    current_file = 0

    for file in os.listdir(real_path):
        current_file += 1
        progress_bar.UpdateBar(current_file, total_files)

        fingerprint_image = cv2.imread(os.path.join(real_path, file), 0)
        sift = cv2.SIFT_create()
        keypoint_1, descriptor_1 = sift.detectAndCompute(sample, None)
        keypoint_2, descriptor_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher().knnMatch(descriptor_1, descriptor_2, k=2)

        match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]
        match_score = len(match_points) / max(len(keypoint_1), len(keypoint_2)) * 100

        if match_score > best_score:
            best_score = match_score
            best_match = (file, fingerprint_image, match_points, keypoint_1, keypoint_2)

    return best_score, best_match

def draw_matches(sample_img, matched_img, match_points, keypoint_1, keypoint_2):
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_GRAY2BGR)

    for match in match_points:
        sample_point = tuple(map(int, keypoint_1[match.queryIdx].pt))
        matched_point = tuple(map(int, keypoint_2[match.trainIdx].pt))
        cv2.circle(sample_img, sample_point, 5, (0, 255, 0), -1)  # Green circle
        cv2.circle(matched_img, matched_point, 5, (0, 255, 0), -1)  # Green circle
        cv2.line(sample_img, sample_point, matched_point, (255, 0, 0), 2)  # Red line

    return np.hstack((sample_img, matched_img))

def main():
    sg.theme('DarkBlue3')

    layout = [
        [sg.Text('Fingerprint Matching Application', font=('Helvetica', 20))],
        [sg.Text('Select Sample Image:'), sg.InputText(key='sample_path'), sg.FileBrowse()],
        [sg.Text('Select Real Images Folder:'), sg.InputText(key='real_path'), sg.FolderBrowse()],
        [sg.Button('Process', size=(10, 2)), sg.Button('Exit', size=(10, 2))],
        [sg.Text('Matched Images with Matching Points:', font=('Helvetica', 12))],
        [sg.Image(key='-IMAGE-')],
        [sg.Text('Progress:'), sg.ProgressBar(100, orientation='h', size=(20, 20), key='progress')],
    ]

    window = sg.Window('Fingerprint Matching', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Process':
            sample_path = values['sample_path']
            real_path = values['real_path']

            if sample_path and real_path:
                progress_bar = window['progress']
                progress_bar.UpdateBar(0, 100)

                best_score, best_match = process_image(sample_path, real_path, progress_bar)
                sg.popup(f'Best Match Score: {best_score:.2f}%')

                if best_match:
                    matched_img_path = os.path.join(real_path, best_match[0])
                    sample_img = cv2.imread(sample_path, 0)
                    matched_img = cv2.imread(matched_img_path, 0)
                    match_points = best_match[2]
                    keypoint_1 = best_match[3]
                    keypoint_2 = best_match[4]

                    # Draw matching points with colors on images
                    matched_output = draw_matches(sample_img, matched_img, match_points, keypoint_1, keypoint_2)

                    # Display matched images with matching points
                    imgbytes = cv2.imencode('.png', matched_output)[1].tobytes()
                    window['-IMAGE-'].update(data=imgbytes)

            else:
                sg.popup_error('Please select both the sample image and real images folder.')

    window.close()

if __name__ == '__main__':
    main()
