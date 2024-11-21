import cv2
import numpy as np
import image_modules

def extract_ellipse(image, center, axes, angle=0, start_angle=0, end_angle=360):
    # create a mask with the same shape as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
 
    # draw the ellipse on the mask
    #cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, (255), -1)
    image_modules.draw_filled_ellipse(mask, center, axes, angle, 255)

    # create an alpha mask
    alpha_mask = image_modules.merge_channels([mask, mask, mask, mask])

    # if the image has 3 channels, convert it to 4 channels
    if image.shape[2] == 3:
        # Create an alpha channel filled with 255 (fully opaque)
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        # Stack the original image with the alpha channel to form a BGRA image
        image = np.dstack([image, alpha_channel])

    # apply the alpha mask to the image
    result = image_modules.bitwise_and(image, alpha_mask)

    return result

# load the image
image_path = 'application_images/image.png' # path to the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) < 2:
    print("There has to be at least two faces in the image.")
else:
    # extract the bounding boxes of the first two faces
    x1, y1, w1, h1 = faces[0]
    x2, y2, w2, h2 = faces[1]
    face1 = image[y1:y1 + h1, x1:x1 + w1]
    face2 = image[y2:y2 + h2, x2:x2 + w2]

    # resize the faces to the same size
    face1_resized = image_modules.resize_image(face1, (w2, h2))
    face2_resized = image_modules.resize_image(face2, (w1, h1))

    # calculate the center and axes of the ellipses
    center1_resized = (w2 // 2, h2 // 2)  # center after resize
    axes1_resized = (w2 // 3, h2 // 2)    # axes after resize
    center2_resized = (w1 // 2, h1 // 2)  # center after resize
    axes2_resized = (w1 // 3, h1 // 2)    # axes after resize

    # extract the ellipses from the resized faces
    face1_ellipse = extract_ellipse(face1_resized, center1_resized, axes1_resized)
    face2_ellipse = extract_ellipse(face2_resized, center2_resized, axes2_resized)

    # create a copy of the original image
    result = image.copy()

    # extract the BGR channels of the ellipses
    face1_resized_bgr = face1_ellipse[:, :, :3]
    face2_resized_bgr = face2_ellipse[:, :, :3]

    # extract the alpha channels of the ellipses
    alpha1 = face1_ellipse[:, :, 3]
    alpha2 = face2_ellipse[:, :, 3]

    # overlay the ellipses on the original image
    for i in range(h2):
        for j in range(w2):
            if face1_ellipse[i, j, 3] != 0:  # only non-transparent pixels
                result[y2 + i, x2 + j, :3] = face1_ellipse[i, j, :3]  # copy BGR values

    for i in range(h1):
        for j in range(w1):
            if face2_ellipse[i, j, 3] != 0:  # only non-transparent pixels
                result[y1 + i, x1 + j, :3] = face2_ellipse[i, j, :3]  # copy BGR values

    # show the result
    cv2.imshow("Face Swap with Ellipses", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
