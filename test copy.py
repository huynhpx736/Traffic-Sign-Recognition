import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
font_size = 0.75
font_path = "ARIAL.TTF"
font = ImageFont.truetype(font_path, int(font_size * 50))
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model=load_model("model.h5")  ## rb = READ BYTE
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if classNo == 0:
        return 'Giới hạn tốc độ (20km/h)'
    elif classNo == 1:
        return 'Giới hạn tốc độ (30km/h)'
    elif classNo == 2:
        return 'Giới hạn tốc độ (50km/h)'
    elif classNo == 3:
        return 'Giới hạn tốc độ (60km/h)'
    elif classNo == 4:
        return 'Giới hạn tốc độ (70km/h)'
    elif classNo == 5:
        return 'Giới hạn tốc độ (80km/h)'
    elif classNo == 6:
        return 'Hết giới hạn tốc độ (80km/h)'
    elif classNo == 7:
        return 'Giới hạn tốc độ (100km/h)'
    elif classNo == 8:
        return 'Giới hạn tốc độ (120km/h)'
    elif classNo == 9:
        return 'Cấm vượt'
    elif classNo == 10:
        return 'Cấm vượt cho xe có trọng lượng trên 3.5 tan'
    elif classNo == 11:
        return 'Ưu tiên tại giao lộ kế tiếp'
    elif classNo == 12:
        return 'Đường ưu tiên'
    elif classNo == 13:
        return 'Nhường đường'
    elif classNo == 14:
        return 'Dừng lại'
    elif classNo == 15:
        return 'Cấm xe cơ giới'
    elif classNo == 16:
        return 'Cấm xe co trọng lượng trên 3.5 tấn'
    elif classNo == 17:
        return 'Cấm đi ngược chiều'
    elif classNo == 18:
        return 'Cảnh báo nguy hiểm'
    elif classNo == 19:
        return 'Đường cong nguy hiểm bên trái'
    elif classNo == 20:
        return 'Đường cong nguy hiểm bên phải'
    elif classNo == 21:
        return 'Đường cong kép'
    elif classNo == 22:
        return 'Duong gập ghềnh'
    elif classNo == 23:
        return 'Đường trơn trượt'
    elif classNo == 24:
        return 'Đường hẹp bên phải'
    elif classNo == 25:
        return 'Công trường đang thi công'
    elif classNo == 26:
        return 'Đèn tín hiệu giao thông'
    elif classNo == 27:
        return 'Cảnh báo người đi bộ'
    elif classNo == 28:
        return 'Trẻ em qua đường'
    elif classNo == 29:
        return 'Giao nhau với xe đạp'
    elif classNo == 30:
        return 'Cảnh báo băng tuyết/đá'
    elif classNo == 31:
        return 'Gặp động vật hoang dã băng qua'
    elif classNo == 32:
        return 'Hết mọi giới hạn tốc độ và cấm vượt'
    elif classNo == 33:
        return 'Rẽ phải ở phía trước'
    elif classNo == 34:
        return 'Rẽ trái ở phía trước'
    elif classNo == 35:
        return 'Chỉ đi thẳng'
    elif classNo == 36:
        return 'Đi thẳng hoặc rẽ phải'
    elif classNo == 37:
        return 'Đi thẳng hoặc rẽ trái'
    elif classNo == 38:
        return 'Giữ bên phải'
    elif classNo == 39:
        return 'Giữ bên trái'
    elif classNo == 40:
        return 'Bắt buộc vòng xuyến'
    elif classNo == 41:
        return 'Hết cấm vượt'
    elif classNo == 42:
        return 'Hết cấm vượt cho xe có trọng lượng trên 3.5 tấn'
 
while True:
    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue =np.amax(predictions)
    # cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # Thay đổi cách vẽ văn bản
    img_PIL = Image.fromarray(imgOrignal)
    draw = ImageDraw.Draw(img_PIL)
    draw.text((20, 35), f"CLASS: {getCalssName(classIndex)}", font=font, fill=(0, 0, 255))
    draw.text((20, 75), f"PROBABILITY: {round(probabilityValue * 100, 2)}%", font=font, fill=(0, 0, 255))

# Chuyển lại sang OpenCV
    imgOrignal = np.array(img_PIL)
    
    cv2.imshow("Result", imgOrignal)

    k=cv2.waitKey(1) 
    if k== ord('q'):
        break

cv2.destroyAllWindows()
cap.release()