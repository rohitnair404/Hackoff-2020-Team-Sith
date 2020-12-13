# importing libraries
import os
import cv2
from PIL import Image

#print(os.getcwd())

os.chdir("D:\\trial\\frames")
path = "D:\\trial\\frames"

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir('.'))
# print(num_of_images)

for file in os.listdir('.'):
    im = Image.open(os.path.join(path, file))
    width, height = im.size
    mean_width += width
    mean_height += height
# im.show() # uncomment this for displaying the image
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)

for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        # opening image using PIL Image
        im = Image.open(os.path.join(path, file))

        # im.size includes the height and width of image
        width, height = im.size
        # resizing
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
        imResize.save(file, 'JPEG', quality=95)  # setting quality
        # printing each resized image name
        #print(im.filename.split('\\')[-1], " is resized")


def generate_video():
    image_folder = '.'  # make sure to use your folder
    video_name = 'mygeneratedvideo.avi'
    os.chdir("D:\\trial\\frames")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


# Calling the generate_video function
generate_video()
