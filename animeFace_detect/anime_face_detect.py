import cv2
import os

def face_detect(file_name, cascade_name, output_dir):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cascade_name)
    faces = face_cascade.detectMultiScale(img_gray)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_count = 0

    for (x, y, w, h) in faces:
        """
        ここは検出の幅の調整
        Y：上に移動
        H：height
        W：width
        調整するには係数を直すこと
        ex. new_y = y - int(h * 1.5)
        """
        new_y = y - int(h * 0.4)
        new_h = int(h * 1.4)
        new_w = int(w * 1.2)


        new_y = max(0, new_y)
        new_h = min(img.shape[0] - new_y, new_h)
        new_w = min(img.shape[1] - x, new_w)


        face_img = img[new_y:new_y + new_h, x:x + new_w]

        # 保存
        face_file_name = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(file_name))[0]}_face_{face_count}.png')
        cv2.imwrite(face_file_name, face_img)
        face_count += 1

        img = cv2.rectangle(img, (x, new_y), (x + new_w, new_y + new_h), (255, 0, 255), 5)

    cv2.imshow('Face detection', img)  # show
    cv2.waitKey(0)  # keep

# batch_size
def batch_process(start_idx, end_idx, cascade_name, input_dir, output_dir):
    for i in range(start_idx, end_idx + 1):
        file_name = os.path.join(input_dir, f'img{i:03d}.png')
        if os.path.exists(file_name):
            face_detect(file_name, cascade_name, output_dir)
        else:
            print(f"File {file_name} does not exist.")


# run
input_dir = 'img_input'
output_dir = 'img_output'
cascade_name = 'lbpcascade_animeface.xml'
batch_process(1, 200, cascade_name, input_dir, output_dir)