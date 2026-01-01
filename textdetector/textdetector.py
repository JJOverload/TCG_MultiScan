# Import for text detection
import cv2 as cv
import math
import argparse
# Import for rectangle
import numpy as np
# Import for rotations (and pytesseract)
from PIL import Image
import pytesseract
#for autocorrect
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
# For jsonparser
import json
# For Timer
import datetime
# For replacing special characters that can cause issues
from module import helper

# Recording start time for timer
starttime = datetime.datetime.now()

# Grabbing arguments from command line when executing command
parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file.')
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

parser.add_argument('--device', default="cpu", help="Device to inference on")

# Location/name of answer .txt file
parser.add_argument('--answername', type=str, default="", help="Location/name of answer .txt file.")

# Indicator of whether or not we show text onto Rec2.jpg
parser.add_argument('--showtext', action='store_true', help="Indicator of whether or not we show text onto Rec2.jpg image.")

args = parser.parse_args()


def convertMTGJSONNamesToVocabAndNames(jsonName):
    names = []
    non_names = [] # Second list of names reserved for things like type info and text of cards 
    unaltered_double_names = []
    separated_double_names = []
    double_temp_list = []
    V = {}
    # AtomicCards_data is a dictionary of dictionaries of MTG card data...
    with open(jsonName, 'r', encoding='utf-8') as AtomicCards_file:
        AtomicCards_data = json.load(AtomicCards_file)
    # creating list/set of names
    names = list(AtomicCards_data["data"].keys()) # First list of names
    #for each "name" in names
    for n in names:
        if len(AtomicCards_data["data"][n]) > 1:
            print("Initially looking at: ", helper.replace_bad_characters(n), "| 'Side' total:", len(AtomicCards_data["data"][n]))
            unaltered_double_names.append(n)
            double_temp_list = split_double_card_names(n)
            for dn in double_temp_list:
                separated_double_names.append(dn)
                print("Appending dn to double_names:", helper.replace_bad_characters(dn))
        for index in range(0, len(AtomicCards_data["data"][n])):
            #print("Looking at: ", n, "| 'Side' number:", index+1)
            print("Looking at: ", helper.replace_bad_characters(n), "| 'Side' number:", index+1)
            #print("-Storing", AtomicCards_data["data"][n][index]["text"], "into non_names...")
            if "text" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["text"]))
            if "type" in AtomicCards_data["data"][n][index]:
                non_names.append(json.dumps(AtomicCards_data["data"][n][index]["type"]))

    # Non-names are also "vocab" we are using. Counting them as "names" for simplicity.
    #print()
    #print(separated_double_names)
    #print()
    for udn in unaltered_double_names:
        names.remove(udn)
    names = separated_double_names + names + non_names
    V = set(names) # V for vocab

    #for vocab in V:
    #    if vocab == "Brazen Borrower":
    #        print("FOUND IT ---------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return(V, names, non_names, separated_double_names, unaltered_double_names)

def split_double_card_names(n):
    if n.find(" // ") != -1:
        return([n[:n.find(" // ")], n[n.find(" // ")+4:]])
    return([n])

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def merge_boxes(box1, box2):
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2

    return( (min(box1_xmin, box2_xmin), min(box1_ymin, box2_ymin), max(box1_xmax, box2_xmax), max(box1_ymax, box2_ymax)) )

def calc_sim(text, obj):
    # text: xmin, ymin, xmax, ymax
    # obj: xmin, ymin, xmax, ymax
    text_xmin, text_ymin, text_xmax, text_ymax = text
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj

    x_dist = min(abs(text_xmin-obj_xmin), abs(text_xmin-obj_xmax), abs(text_xmax-obj_xmin), abs(text_xmax-obj_xmax))
    y_dist = min(abs(text_ymin-obj_ymin), abs(text_ymin-obj_ymax), abs(text_ymax-obj_ymin), abs(text_ymax-obj_ymax))

    dist = x_dist + y_dist

    return(dist)

def compare_vertices_with_box_shows_overlap(vertices, boxtemp):
    index = 0
    while (index < len(vertices)):
        x1, y1 = vertices[index] #x1, y1 is a temp vertice we are looking at right now from vertices
        # text: xmin, ymin, xmax, ymax
        # obj: xmin, ymin, xmax, ymax
        if (x1 >= boxtemp[0]) and (x1 <= boxtemp[2]) and (y1 >= boxtemp[1]) and (y1 <= boxtemp[3]):
            return(True)
        index = index + 1
    return(False)

def is_overlap(text, obj):
    text_xmin, text_ymin, text_xmax, text_ymax = text
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj

    textvertices = [(text_xmin, text_ymin), (text_xmax, text_ymin), (text_xmin, text_ymax), (text_xmax, text_ymax)]
    objvertices = [(obj_xmin, obj_ymin), (obj_xmax, obj_ymin), (obj_xmin, obj_ymax), (obj_xmax, obj_ymax)]

    if (compare_vertices_with_box_shows_overlap(textvertices, obj) or compare_vertices_with_box_shows_overlap(objvertices, text)):
        #if compare is true (you found a vertex that is within the bbox of the other list of tuples)
        return(True)
    return(False)


def merge_algo(bboxes): #bboxes is a list of bounding boxes data
    for j in bboxes:
        for k in bboxes:
            if j == k: #continue on if we are comparing a box with itself
                continue
            # Find out if these two bboxes are within distance limit
            if (calc_sim(j, k) < dist_limit) or (is_overlap(j, k) == True):
                # Create a new box
                new_box = merge_boxes(j, k)
                bboxes.append(new_box)
                # Remove previous boxes
                bboxes.remove(j)
                bboxes.remove(k)

                #Return True and new "bboxes"
                return(True, bboxes)
    return(False, bboxes)

#Finding Similar Names for autocorrector portion
def mtg_autocorrect(input_word, V, name_freq_dict, probs):
    #input_word = input_word.lower()

    # qval in similarities needs to be 2, meaning input_word needs to be 2 characters or more.
    if len(input_word) == 2:
        input_word = input_word + " "
    if len(input_word) == 1:
        input_word = input_word + "  "
    elif len(input_word) == 0:
        input_word = input_word + "   "
    similarities = [1-(textdistance.Jaccard(qval=3).distance(v,input_word)) for v in name_freq_dict.keys()]
    #similarities = [(textdistance.Sorensen(qval=2).similarity(v,input_word)) for v in name_freq_dict.keys()]
    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
    df = df.rename(columns={'index':'Name', 0:'Prob'})
    df['Similarity'] = similarities
    output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(1) #.iat[0,0]

    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #tempoutput = df.sort_values(['Similarity', 'Prob'], ascending=False).head(20) #.iat[0,0]
    #print(tempoutput)
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    #print("output:\n", output)
    #if output.iat[0,2] <= 0.1:
    #    return("")
    return(output)

#Converting txt file true list for accuracy checker.
#Input: Text file name or location
#Output: List
def convertTxtFileAnswerToList(aname):
    outputlist = []
    f = open(aname, "r")
    for x in f: #x is a line of the text file
        outputlist.append(x.strip())
        print("During conversion from answer file -> list:|" + str(x.strip()) + "|")
    f.close()
    return(outputlist)

#Finding out the accuracy of results compared to answer list
def runAccuracyChecker(bestNameListNameOnly, answerList):
    correctcounter = 0
    incorrectcounter = 0
    leftovercounter = 0

    print("Length of bestNameListNameOnly: " + str(len(bestNameListNameOnly)))
    print("Length of answerList: " + str(len(answerList)))

    #OCR & autocorrector result list (bestNameListNameOnly) might be less in length compared to answerList
    if len(bestNameListNameOnly) < len(answerList):
        #leftover names not mentioned in result list counts as incorrect as well.
        leftovercounter = len(answerList) - len(bestNameListNameOnly)

    #checking each best name against answerList
    for nameofb in bestNameListNameOnly:
        if nameofb in answerList:
            correctcounter = correctcounter + 1
            print("(Correct): " + str(nameofb))
            answerList.remove(nameofb)
        else: #incorrect
            incorrectcounter = incorrectcounter + 1
            print("***(NOT Correct)***: " + str(nameofb))

    
    print("---------------------------------------------")
    print("Remaining names in answer list (not guessed/not guessed correctly):", answerList)
    print("---------------------------------------------")
    print("Correct Guess Score (correctcounter): " + str(correctcounter))
    print("Incorrect Guess Penalty (incorrectcounter): " + str(incorrectcounter))
    print("Leftover Penalty (leftovercounter, if answer list > result list): " + str(leftovercounter))
    print("Accuracy of Results Against Answer (correct/(correct+incorrect+leftover): " + str(correctcounter/(correctcounter+incorrectcounter+leftovercounter)))

def find_good_thresh(cap):
    rows, cols = cap.shape
    string = ""
    lowest = 500
    highest = -1
    for y in range(0, rows):
        for x in range(0, cols):
            if cap[y][x] > highest:
                highest = cap[y][x]
            if cap[y][x] < lowest:
                lowest = cap[y][x]
    
    int_result = int((int(lowest)+int(highest))/2)
    print("Lowest: " + str(lowest))
    print("Highest: " + str(highest))
    print("(Lowest+Highest)/2: " + str(int_result))

    return(int_result)

if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    answerfilename = args.answername
    showthetext = args.showtext

    # Creating buffer/container for bounding boxes' vertices
    bbox = []
    # Setting distance limit for bounding box merging
    dist_limit = 40
    # distance for expanding bounding box.
    # Combined values of expanding distances should not exceed dist_limit. dist_limit/2 can likely cause decrease in accuracy
    expand_dist = int(dist_limit/10)
    #expand_dist = 0

    # Load network
    net = cv.dnn.readNet(model)
    if args.device == "cpu":
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu": #not tested
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    # Create a new named window
    #kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    #cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
    #cap = cv.VideoCapture(args.input if args.input else 0)
    #cap = cv.VideoCapture(args.input)
    cap = cv.imread(args.input)
    #cap = cv.imread(args.input, cv.IMREAD_GRAYSCALE)

    

    print("-----Opening Atomic Cards JSON------")
    V, names, non_names, separated_double_names, unaltered_double_names = convertMTGJSONNamesToVocabAndNames("AtomicCards.json")
    
    
    # Counter of name frequency
    name_freq_dict = {}
    name_freq_dict = Counter(names)
    #Relative Frequency of names
    print("-----Populating dictionary of Name Frequencies-----")
    probs = {}
    Total = sum(name_freq_dict.values())
    for k in name_freq_dict.keys():
        probs[k] = name_freq_dict[k]/Total


    print("-----Starting Reading of Image------")
    #while cv.waitKey(1) < 0:
    # Read frame
    #hasFrame, frame = cap.read()
    frame = cap

    #if not hasFrame:
        #cv.waitKey()
        #break

    print("-----Getting Frame Height and Width------")
    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    print("-----Create a 4D Blob from Frame------")
    # Create a 4D blob from frame.
    blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    print("-----Running the Model------")
    # Run the model
    net.setInput(blob)
    output = net.forward(outputLayers)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    
    # have buffers for getting coordinates for rectangles
    startCorner = (0, 0)
    endCorner = (0, 0)
    #creating mask layer
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    #masked = None
    mask2 = np.zeros(frame.shape[:2], dtype="uint8")

    #creating backup frame
    frame2 = frame.copy()

    print("-----Getting the Scores and Geometry------")
    # Get scores and geometry
    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    # Apply NMS
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold) 
    for i in indices:
        wlist = []
        hlist = []
        # get 4 corners of the rotated rect
        vertices = cv.boxPoints(boxes[i])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
            #populating list for min/max getting and text isolation
            wlist.append(int(vertices[j][0]))
            hlist.append(int(vertices[j][1]))
            #print("Appended:", (int(vertices[j][0]), int(vertices[j][1]) ) )
        

        #print("-----Initial vertices for a box completed-----")
        # text: ymin, xmin, ymax, xmax
        # obj: ymin, xmin, ymax, xmax
        # order of parameters currently not synchronized with initial algorithm

        #integrating expanding distance to list of bounding box values
        xmin, ymin, xmax, ymax = min(wlist)-expand_dist, min(hlist)-expand_dist, max(wlist)+expand_dist, max(hlist)+expand_dist
        bbox.append((xmin, ymin, xmax, ymax))
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            p1 = (int(p1[0]), int(p1[1]))
            #print("p1:",p1)
            p2 = (int(p2[0]), int(p2[1]))
            #print("p2:",p2)

            # Drawing line
            cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
            #cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            # Making rectangle and then applying it as a mask
            cv.rectangle(mask, (min(wlist)-expand_dist, min(hlist)-expand_dist), (max(wlist)+expand_dist, max(hlist)+expand_dist), 255, -1)
            masked = cv.bitwise_and(frame2, frame2, mask=mask)
    # Put efficiency information
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    

    #print("-----Printing Out Initial \"bboxes\"-----:")
    #for b in bbox:
    #    print(b)

    print("-----Merging Initial \"bboxes\"-----:")
    #Merge the boxes
    need_to_merge = True
    while need_to_merge:
        need_to_merge, bbox = merge_algo(bbox)

    print("-----Merging Done. Initializing Variables-----:")
    #initialization for optimization
    bestNameList = [] #for best name list and similarity values
    bestNameListNameOnly = []
    noiseList = []
    counter = 0
    mask3 = None
    masked3 = None
    path = ""
    image_to_be_cropped = None
    cropped_image = None
    counter2 = 0
    rotatelist = []
    masked3_initial_grayscale = None
    thresh = None
    masked3_black_and_white = None
    sharpened_black_and_white = None
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) #don't change to None
    path_black_white = ""

    #Initialization of variables
    masked3_copy = None
    masked3_rotated = None
    #Initialization of variables
    degree = 0
    masked3_rotated = None
    imageToStrStr = None
    autocorrectOutput = None
    tempSimilarity = 0.0
    maxSimilarity = 0.0
    startcountdown = False
    bestOutput = None
    initial_bestOutput = mtg_autocorrect("", V, name_freq_dict, probs)
    countdowncounter = 0
    
    print("-----Iterating Through Merged \"bboxes\"-----:")
    for b in bbox:
        counter += 1
        print("->", counter, b)
        # text: xmin, ymin, xmax, ymax
        # obj: xmin, ymin, xmax, ymax
        #Making rectangles for mask2, which will be be used to ultimately generate "Rec2.jpg"
        cv.rectangle(mask2, (b[0], b[1]), (b[2], b[3]), 255, -1)
        
        # Creating mask3 to create box image in "box_images" directory
        mask3 = np.zeros(frame2.shape[:2], dtype="uint8")
        cv.rectangle(mask3, (b[0], b[1]), (b[2], b[3]), 255, -1)
        masked3 = cv.bitwise_and(frame2, frame2, mask=mask3)

        # Likely would need to modify this line below if using Linux. Use this line to help with debugging. Would need to create box_images directory first.
        path = ".\\box_images\\box"+str(counter)+".jpg"
        cv.imwrite(path, masked3)

        # The coordinates for the cropping box are (left, upper, right, lower)
        image_to_be_cropped = Image.open(path)
        box = (b[0], b[1], b[2], b[3])
        cropped_image = image_to_be_cropped.crop(box)
        path2 = ".\\box_images\\box"+str(counter)+"_"+"cropped"+".jpg"
        cropped_image.save(path2)

        # Hackish way to make image have more "details" within pixels
        # by making use of the innate compression settings of .imwrite method
        image = cv.imread(path2)
        temppath = ".\\box_images\\temp_image"+str(counter)+".jpg"
        cv.imwrite(temppath , image)

        # Saving variations of frames in rotations
        counter2 = 0
        rotatelist = [6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6]
        
        # Grayscale/BnW coding and then sharpen here
        #masked3_initial_grayscale = cv.imread(path2, cv.IMREAD_GRAYSCALE)
        #thresh, masked3_black_and_white = cv.threshold(masked3_initial_grayscale, find_good_thresh(masked3_initial_grayscale), 255, cv.THRESH_BINARY) #thresh is a dummy value
        #masked3_black_and_white = cv.merge((masked3_black_and_white, masked3_black_and_white, masked3_black_and_white))
        
        # Create the sharpening kernel (already initialized)
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        # Sharpen the image 
        #sharpened_black_and_white = cv.filter2D(masked3_black_and_white, -1, kernel) 

        #temp test
        #masked3_initial = cv.imread(path2)
        #sharpened_black_and_white = cv.filter2D(masked3_initial, -1, kernel) 

        #path_black_white = ".\\box_images\\box"+str(counter)+"_"+"sharpen-BnW"+".jpg"
        #cv.imwrite(path_black_white, masked3_black_and_white)
        #cv.imwrite(path_black_white, sharpened_black_and_white)

        # Rotational coding here
        # Assigning default variable values here
        #masked3_copy = Image.open(path_black_white)
        masked3_copy = Image.open(temppath)
        masked3_rotated = None
        maxSimilarity = 0.0
        bestOutput = initial_bestOutput
        startcountdown = False
        countdowncounter = 0
        i = 0 #index for rotatelist
        while(i < len(rotatelist)):
            degree = rotatelist[i]
            counter2 += 1
            masked3_rotated = masked3_copy.rotate(degree)

            path3 = ".\\box_images\\box"+str(counter)+"_"+str(counter2)+".jpg"
            masked3_rotated.save(path3)
            imageToStrStr = pytesseract.image_to_string(masked3_rotated)
            imageToStrStr = imageToStrStr.strip() #removing leading and trailing newlines/whitespace
            autocorrectOutput = mtg_autocorrect(imageToStrStr, V, name_freq_dict, probs)
            tempSimilarity = autocorrectOutput.iat[0,2]
            if tempSimilarity > maxSimilarity:
                maxSimilarity = tempSimilarity
                bestOutput = autocorrectOutput
                countdowncounter = 0 #reset countdown counter if max similarity is updated
            if maxSimilarity >= 0.6:
                startcountdown = True
            if startcountdown == True:
                countdowncounter = countdowncounter + 1
            print("countdowncounter: " + str(countdowncounter))
            print(path3 + ": '" + str(imageToStrStr) + "'\n" + str(autocorrectOutput))
            if countdowncounter >= 3:
                print("~Skipping rest of rotations~")
                break
            i = i + 1
            print("---------------------------------------------")
        print("Best Name:\n" + str(bestOutput)) #output the "best" name extracted from among all the rotated images for this bounding box
        if (bestOutput.iat[0,0] in non_names) or (bestOutput.iat[0,2] <= 0.30):
            noiseList.append((bestOutput.iat[0,0], bestOutput.iat[0,2], b, counter))
            print("-----Likely Noise/Non-name - Skipped-----")
        else: #If name is exising or is not "noise" due to low similarity
            bestNameList.append((bestOutput.iat[0,0], bestOutput.iat[0,2], b, counter))
            print("---------------------------------------------")
        print("")


    print("-----Outputing Best Name List-----")
    print(bestNameList)
    print("Length of bestNameList: " + str(len(bestNameList)))
    for n, s, box, c in bestNameList:
        print(n)
        bestNameListNameOnly.append(n)

    #merging frame2 and mask2 to make masked2 altered frame
    masked2 = cv.bitwise_and(frame2, frame2, mask=mask2)
    masked2copy = cv.bitwise_and(frame2, frame2, mask=mask2)
    
    #edit masked2copy if "--showtext" argument is specified/true
    if (showthetext == True):
        for n, s, box, c in bestNameList:
            #cv.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv.putText(masked2copy, str(str(c)+": "+n), (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        for n, s, box, c in noiseList:
            cv.putText(masked2copy, str(str(c)+": ~Noise~"), (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    #execute AccuracyChecker, if argument is present in run command
    if (answerfilename != ""):
        answerList = convertTxtFileAnswerToList(answerfilename)
        runAccuracyChecker(bestNameListNameOnly, answerList)


    cv.imwrite("output.png", frame)
    cv.imwrite("Rec.jpg", masked)
    cv.imwrite("Rec2.jpg", masked2)
    cv.imwrite("Rec3.jpg", masked2copy)
    
    
    #Recording endtime and outputing elapsed time
    endtime = datetime.datetime.now()
    elapsedtime = endtime - starttime
    print("Elapsed Time:", elapsedtime)

    '''
    ####
    # Name of window
    kWinName = "TCG MultiScan"
    # Spawn window
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    # Display the frame
    cv.imshow(kWinName, masked2copy)
    cv.waitKey()
    ####
    '''



