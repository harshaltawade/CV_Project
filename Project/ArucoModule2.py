import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    # Your existing code for loading marker images
    # ...
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of Markers Detected: ",noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def loadAugVideos(path):
    """
    Load video files for each marker ID.
    :param path: Folder where video files are stored.
    :return: Dictionary with key as the ID and values as VideoCapture objects.
    """
    myList = os.listdir(path)
    augVideos = {}
    for videoPath in myList:
        key = int(os.path.splitext(videoPath)[0])
        cap = cv2.VideoCapture(f'{path}/{videoPath}')
        augVideos[key] = cap
    return augVideos

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    # Your existing code for marker detection
    # ...
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs,ids, rejected = aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)

    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    Augment the ArUco marker with an image or video while preserving the background.
    :param bbox: Four corner points of the marker's bounding box.
    :param id: Marker ID.
    :param img: Image on which to draw.
    :param imgAug: The image or video to overlay on the marker.
    :param drawId: Flag to display the marker ID.
    :return: Image with the augmented image or video.
    """
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    matrix, _ = cv2.findHomography(pts2, pts1)
    warp_img = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Create a mask to blend the augmented content with the original image
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts1.astype(int), (255, 255, 255))
    inv_mask = cv2.bitwise_not(mask)

    imgOut = cv2.bitwise_and(img, inv_mask)
    imgOut = cv2.bitwise_or(imgOut, warp_img)

    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut


def main():
    cap = cv2.VideoCapture(0)
    augDics = loadAugImages("Markers")
    augVideos = loadAugVideos("Videos")  # Load videos for augmentation

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Loop through all the markers and augment each one
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])
                elif int(id) in augVideos.keys():
                    video_capture = augVideos[int(id)]
                    ret, frame = video_capture.read()
                    if ret:
                        img = augmentAruco(bbox, id, img, frame)
                    else:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning

        cv2.imshow("Image", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    for video_capture in augVideos.values():
        video_capture.release()  # Release video capture objects

if __name__ == "__main__":
    main()
