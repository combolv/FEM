import cv2

out_path = ".\\out2\\"
video_size = (432, 768)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("out_test.mp4", fourcc, 30, video_size, True)
for i in range(300):
    img = cv2.imread(out_path + str(i).zfill(4) + ".png")
    video.write(img)

video.release()