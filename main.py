import cv2
import pathlib
import argparse
import json

from termcolor import colored
from yolov4 import YoloV4
from SeamCarving import SeamCarving

def main(path_in, path_out, c_rate=None, r_rate=None, threshold=0.85, jsonfile=None):
    # Predict Objects
    img = cv2.imread(path_in)
    
    # Predict Objects
    yolov4 = YoloV4(threshold=0.85)
    BBOXs = yolov4.getBBOXs(img.copy(), visualize=True)
    bboxs = BBOXs["bboxs"]

    if jsonfile != None:
        print(f"(!) Detected {colored(jsonfile, 'yellow')}, extended BBOXs")
        with open(jsonfile, 'r') as f:
            bboxs.extend(json.load(f)["bboxs"])
        
    print("Json BBOXs", colored(json.dumps(BBOXs), 'green'))

    # Start Seam Carving
    SC = SeamCarving(bboxs=bboxs)
    
    if c_rate != None:
        img = SC.crop_c(img, c_rate)
    if r_rate != None:
        img = SC.crop_r(img, r_rate)
    
    cv2.imwrite(path_out, img)
    cv2.destroyAllWindows()

def load_bboxs(filename):
    assert pathlib.Path(args.input).exists() and pathlib.Path(args.file).exists()

if __name__ == '__main__':
    doc_parameters = f'''
    @ parameters:
    
        "-i", "--input"     , required=True,                                type=str        help="input image path"
        "-o", "--output"    , required=False, default="results/output.png", type=str,       help="output image path"
        "-c", "--c_rate"    , required=False, default=None,                 type=float,     help="column cropping rate"
        "-r", "--r_rate"    , required=False, default=None,                 type=float,     help="row cropping rate"
        "-t", "--threshold" , required=False, default=0.85,                 type=float,     help="predict threshold"
        "-f", "--file"      , required=False, default=None,                 type=str,       help="JSON notations file"
    '''

    doc_info = f'''
    @ author  : Burak Büyükyüksel
    @ license : Public
    @ Version : 2.0.0

    '''

    print(colored(doc_parameters, "cyan"))
    print(colored(doc_info, "green"))

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input" , required=True,  help="input image path")
    ap.add_argument("-o", "--output", required=False, default="results/output.png", help="output image path")
    ap.add_argument("-c", "--c_rate", required=False, default=None, type=float, help="column cropping rate")
    ap.add_argument("-r", "--r_rate", required=False, default=None, type=float, help="row cropping rate")
    ap.add_argument("-t", "--threshold", required=False, default=0.85, type=float, help="predict threshold")
    ap.add_argument("-f", "--file", required=False, default=None, help="JSON notations file")
    args = ap.parse_args()

    if args.file != None:
        assert pathlib.Path(args.file).exists(), "Please enter a correct json notations file path!"
    
    print("input    :", colored(args.input, "cyan"))
    print("output   :", colored(args.output, "cyan"))
    print("c_rate   :", colored(args.c_rate, "cyan"))
    print("r_rate   :", colored(args.r_rate, "cyan"))
    print("threshold:", colored(args.threshold, "cyan"))
    
    assert args.c_rate != None or args.r_rate != None, "Please specify at least '1' rate value!"

    assert pathlib.Path(args.input).exists(), "Please enter a correct input image path!"

    # Create Output Directory
    output = pathlib.Path(args.output)
    if not output.parent.exists():
        print("(!) Creating directory ", colored(output.parent, "yellow"))
        output.parent.mkdir()
    
    main(path_in=args.input, path_out=args.output, c_rate=args.c_rate, r_rate=args.r_rate, threshold=args.threshold, jsonfile=args.file)