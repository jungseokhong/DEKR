import cv2
import numpy as np
import os


folders = ['MF_pose1_water']

# Load images
# src_folder = args.srcfolder

for folder in folders:
    src_folder = os.path.join('new_data_011423',folder)

    isExist = os.path.exists(os.path.join('new_data_011423',folder+'_cropped'))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join('new_data_011423',folder+'_cropped'))

    for filename in os.listdir(src_folder):
        img_number = int(filename[3:-4])
        img = cv2.imread(os.path.join(src_folder,filename))
        frame_height, frame_width, _ = img.shape

        if True: #img_number < 587:
            cropped_image = img[:, :round(frame_width*0.70)]
            cv2.imwrite(os.path.join('new_data_011423',folder+'_cropped',filename), cropped_image)
        # elif img_number >= 400 and img_number < 1300:
            # cropped_image = img[:, round(frame_width*0.15):]
            # cv2.imwrite(os.path.join('new_data_011423',folder+'_cropped',filename), cropped_image)
        # elif img_number >= 587 and img_number < 872:
        #     cropped_image = img[:, :round(frame_width*0.9)]
        #     cv2.imwrite(os.path.join('new_data_011423',folder+'_cropped',filename), cropped_image)
        # elif img_number >= 872 and img_number < 1100:
        #     cropped_image = img[:, :round(frame_width*0.70)]
        #     cv2.imwrite(os.path.join('new_data_011423',folder+'_cropped',filename), cropped_image)
        # elif img_number >= 1100:
        #     cropped_image = img[:, :round(frame_width*0.60)]
        #     cv2.imwrite(os.path.join('new_data_011423',folder+'_cropped',filename), cropped_image)
        # # cv2.imwrite("Cropped Image.jpg", cropped_image)



# cv2.imshow("original", img)
 
# # Cropping an image
# cropped_image = img[80:280, 150:330]


# cv2.imwrite("Cropped Image.jpg", cropped_image)
