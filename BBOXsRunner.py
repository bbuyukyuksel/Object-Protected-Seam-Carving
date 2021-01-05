import cv2
import json
import termcolor
import datetime
import argparse
import pathlib

def main(img_path, file_path, d_time=1000):
    img = cv2.imread(img_path)
    
    while True:
        t = img.copy()
        try:
            with open(file_path, 'r') as f:
                BBOXs = json.load(f)["bboxs"]
            for bbox in BBOXs:
                x, y, w, h = bbox
                cv2.rectangle(t, (x,y), (x+w, y+h), (0,255,0), 1)
            cv2.imshow("Live-BBOXs", t)
            key = cv2.waitKey(d_time)
            if (key & 0xFF) == 27:
                break

            print(termcolor.colored(datetime.datetime.now(), "green"), "updated BBOXs", end='\r')
        except:
            print(termcolor.colored("(!) Lütfen JSON formatının doğru olduğundan emin olun.", "red"), "\n")

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input" , required=True,  help="input image path")
    ap.add_argument("-f", "--file", required=True, help="JSON notations file")
    ap.add_argument("-t", "--time", required=False, default=1000, type=int, help="Update Time")

    args = ap.parse_args()
    
    assert pathlib.Path(args.input).exists() and pathlib.Path(args.file).exists()

    main(args.input, args.file, d_time=args.time)
