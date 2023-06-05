import os
import cv2
import numpy as np


def are_open(dict_cap):
    for fp in dict_cap: 
        if not dict_cap[fp].isOpened(): 
            return False
    return True


def release(dict_cap):
    for fp in dict_cap: 
        dict_cap[fp].release()


def read_frames(dict_cap):
    frames = {}
    for fp in dict_cap:
        ret, frame = dict_cap[fp].read()
        if ret == False: 
            return False, None
        frames[fp] = frame
    return True, frames


def main(name, stream_type="video"):

    video_fp_vec = [
        "{}_trtpose.mp4".format(name),
        "{}_openpose.mp4".format(name),
        "{}_parcopose.mp4".format(name),
        "{}_parcopose_h36m.mp4".format(name),
        "{}_parcopose_h36m_openpose.mp4".format(name),
        "{}_parcopose_h36m_CPN.mp4".format(name),
    ]

    label_video_fp = {
        "{}_trtpose.mp4".format(name): "TRTPose",
        "{}_openpose.mp4".format(name): "OpenPose",
        "{}_parcopose.mp4".format(name): "ParcoPose",
        "{}_parcopose_h36m.mp4".format(name): "ParcoPose retrained H3.6M",
        "{}_parcopose_h36m_openpose.mp4".format(name): "ParcoPose retrained H3.6M teacher OpenPose",
        "{}_parcopose_h36m_CPN.mp4".format(name): "ParcoPose retrained H3.6M teacher CPN",
    }
    caps = {fp: cv2.VideoCapture(fp) for fp in video_fp_vec}

    if not are_open(caps): 
        print("Error opening video stream or file")
    
    i = 0
    out_stream = None
    while are_open(caps): 
        ret, frames = read_frames(caps)
        if ret == True: 
            for fp in frames:
                # Put Label
                w, h = frames[fp].shape[1], frames[fp].shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 3
                text = "{}".format(label_video_fp[fp])
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.putText(frames[fp], text, (int(w/2 - text_size[0]/2) + 5, h - text_size[1]), font, font_scale, (255, 255, 255), font_thickness)
                

            frames_list = list(frames.values())

            image_row0 = cv2.hconcat([frames_list[0], frames_list[1], frames_list[2]])
            image_row1 = cv2.hconcat([frames_list[3], frames_list[4], frames_list[5]])
            # image_row1_empty = np.zeros((image_row1.shape[0], image_row1.shape[1] + frames_list[2].shape[1], image_row1.shape[2]), np.uint8)
            # image_row1_empty[:, frames_list[3].shape[1]//2:frames_list[2].shape[1]//2+image_row1.shape[1], :] = image_row1
            # image_row1 = image_row1_empty
            image = cv2.vconcat([image_row0, image_row1])

            if stream_type == "folder":
                if i == 0:
                    folder_name = name + "_comparison"
                    if not os.path.isdir(folder_name):
                        os.mkdir(folder_name)
                cv2.imwrite(os.path.join(folder_name, "{}.png".format(i)), image)
            else: 
                if i == 0:
                    filename = "{}.mp4".format(name + "_comparison")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    w, h = image.shape[1], image.shape[0]
                    out_stream = cv2.VideoWriter(filename, fourcc, 50.0, (w, h))
                if out_stream:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    out_stream.write(image)
            print(i, end="\r")
            i += 1
        else: 
            print("End")
            break
    release(caps)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Concat", epilog="PARCO")
    parser.add_argument("--name", 
                        "-n", 
                        dest="name", 
                        required=True, 
                        help="Model name (parcopose or trtpose)")
    args = parser.parse_args()
    main(args.name, "video")
    # main(args.name, "folder")