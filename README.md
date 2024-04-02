# Age-and-prediction-models
나이 및 성별 예측 모델 -1
<br>
<br>
<br>

# 블로그 포스팅(자세한)
https://junlove-dam1ary.tistory.com/103
<br>
<br>
<br>

# [ 나이 및 성별 인종 예측 결과물 ]
<br>

### 무작위 사진을 가지고 결과를 확인해 보겠습니다.
<pre>
  from keras.utils import to_categorical
from PIL import Image

class UtkFaceDataGenerator():
    def __init__(self, df):
        self.df = df

    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])
        self.max_age = self.df['age'].max()

        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path):
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        return im

    def generate_images(self, image_idx, is_training, batch_size=16):
        images, ages, races, genders = [], [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]

                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']

                im = self.preprocess_image(file)

                ages.append(age / self.max_age)
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)

                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    images, ages, races, genders = [], [], [], []

            if not is_training:
                break
</pre>
<br>
<br>
<pre>
  import matplotlib.pyplot as plt
  
  ### 데이터 제너레이터 생성
  data_generator = UtkFaceDataGenerator(df)
  
  ### 학습 데이터 생성
  train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
  train_images, train_labels = next(data_generator.generate_images(train_idx, is_training=True, batch_size=16))
  
  ### 생성된 이미지 출력
  plt.figure(figsize=(10, 10))
  for i in range(12):
      plt.subplot(4, 3, i+1)
      plt.imshow(train_images[i])
      plt.title(f"Age: {train_labels[0][i] * data_generator.max_age}, Race: {np.argmax(train_labels[1][i])}, Gender: {np.argmax(train_labels[2][i])}")
      plt.axis('off')
  
  plt.tight_layout()
  plt.show()
</pre>
<br>
<br>

### 아래와 같은 결과를 볼 수 있습니다. 
![image](https://github.com/dlwnsgur9242/Age-and-prediction-models/assets/90494150/01b0ab60-678f-4a3c-80f0-8168865a4433)
<br>
<br>
<br>

### 숫자료 표현 돼있는 데이터를 문자열로 변환해 주었습니다.
<pre>
  # 필요한 라이브러리 IMPORT
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import seaborn as sns
import plotly.graph_objects as go

# 숫자로 표현되어있는 인종, 성별 데이터를 문자열로 변환해주기 위한 dictionary 생성
dataset_dict = {
    'race_id': {
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())

folder_name = 'UTKFace'

# 데이터 프레임으로 만들어주기 위한 함수 정의
def parse_dataset(dataset_path, ext='jpg'):
    def parse_info_from_file(path):
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob(os.path.join(dataset_path, "*.%s" % ext))
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    return df

# UTKFace 데이터셋을 데이터 프레임으로 변환
df = parse_dataset(folder_name)

# 랜덤으로 16개 이미지 선택
random_images = df.sample(n=12)

# 선택한 이미지 출력
plt.figure(figsize=(12, 8))
for i, (index, row) in enumerate(random_images.iterrows(), start=1):
    img = cv2.imread(row['file'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(4, 3, i)
    plt.imshow(img)
    plt.title(f"Age: {row['age']}, Gender: {row['gender']}, Race: {row['race']}")
    plt.axis('off')

plt.show()
</pre>
![image](https://github.com/dlwnsgur9242/Age-and-prediction-models/assets/90494150/2569c396-5018-4e8d-9fc3-a45e3d87e6e8)
<br>
<br>
<br>

### 1. 사진으로 나이, 성별 예측하기
### 결과 1)
<pre>
  import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 모델 파일 경로
gender_model_path = './gender_model.keras'
age_model_path = "./age_model.keras"

# 모델 로드
age_model = load_model(age_model_path, compile=False)
gender_model = load_model(gender_model_path, compile=False)

# OPENCV Cascadeclassifier 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 처리
image_size = 200
pic = cv2.imread('./ShinSe-kyung.png')
faces = face_cascade.detectMultiScale(pic, scaleFactor=1.11, minNeighbors=8)

# 결과를 저장할 리스트 초기화
age_ = []
gender_ = []

# 얼굴마다 예측 수행
for (x, y, w, h) in faces:
    img = pic[y:y + h, x:x + w]
    img = cv2.resize(img, (image_size, image_size))
    age_predict = age_model.predict(np.array([img]))
    gender_predict = gender_model.predict(np.array([img]))

    age_.append(age_predict[0][0])
    gender_.append(np.round(gender_predict[0][0]))

    # 결과를 이미지에 표시
    gend = np.round(gender_predict[0][0])
    if gend == 0:
        gend_str = 'Man'
        col = (255, 255, 0)
    else:
        gend_str = 'Woman'
        col = (203, 12, 255)
    cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 225, 0), 1)
    cv2.putText(pic, "Age:" + str(int(age_predict[0][0])) + " / " + str(gend_str), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, w * 0.005, col, 1)

# 이미지 출력
pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 16))
print(age_, gender_)
plt.imshow(pic1)
plt.show()
</pre>
![image](https://github.com/dlwnsgur9242/Age-and-prediction-models/assets/90494150/b7bc902f-33ee-4fef-9b7e-3f69baeb1fc4)
<br>
<br>

### 결과 2)
<pre>
  import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 모델 파일 경로
gender_model_path = './gender_model.keras'
age_model_path = "./age_model.keras"

# 모델 로드
age_model = load_model(age_model_path, compile=False)
gender_model = load_model(gender_model_path, compile=False)

# OPENCV Cascadeclassifier 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 처리
image_size = 200
pic = cv2.imread('./chaseung-won.png')
faces = face_cascade.detectMultiScale(pic, scaleFactor=1.11, minNeighbors=8)

# 결과를 저장할 리스트 초기화
age_ = []
gender_ = []

# 얼굴마다 예측 수행
for (x, y, w, h) in faces:
    img = pic[y:y + h, x:x + w]
    img = cv2.resize(img, (image_size, image_size))
    age_predict = age_model.predict(np.array([img]))
    gender_predict = gender_model.predict(np.array([img]))

    age_.append(age_predict[0][0])
    gender_.append(np.round(gender_predict[0][0]))

    # 결과를 이미지에 표시
    gend = np.round(gender_predict[0][0])
    if gend == 0:
        gend_str = 'Man'
        col = (255, 255, 0)
    else:
        gend_str = 'Woman'
        col = (203, 12, 255)
    cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 225, 0), 1)
    cv2.putText(pic, "Age:" + str(int(age_predict[0][0])) + " / " + str(gend_str), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, w * 0.005, col, 1)

# 이미지 출력
pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 16))
print(age_, gender_)
plt.imshow(pic1)
plt.show()
</pre>
![image](https://github.com/dlwnsgur9242/Age-and-prediction-models/assets/90494150/d74fd418-f9b0-4299-a8a0-e543153237f9)
<br>
<br>
<br>

### 2. 실시간 캠으로 나이 및 성별 예측하기
<pre>
  import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 웹캠 불러오기
camera = cv2.VideoCapture(0)

# 모델 로드
age_model_path = "./age_model.keras"
gender_model_path = './gender_model.keras'
age_model = load_model(age_model_path)
age_model.compile(loss='mse', optimizer='adam')  # 손실 함수 명시
gender_model = load_model(gender_model_path)
gender_model.compile(loss='mse', optimizer='adam')  # 손실 함수 명시


# 얼굴 검출을 위한 OPENCV Cascade Classifier load
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

image_size = 200

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = camera.read()

    # 프레임에서 얼굴 검출
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.11, minNeighbors=8)

    for (x, y, w, h) in faces:
        # 얼굴 이미지 추출 및 크기 조정
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (image_size, image_size))

        # 모델에 입력 전에 이미지를 정규화
        age_predict = age_model.predict(np.array(face_img).reshape(-1, image_size, image_size, 3) / 255.0)
        gender_predict = gender_model.predict(np.array(face_img).reshape(-1, image_size, image_size, 3) / 255.0)

        # 성별 예측값을 문자열로 변환
        gender_str = "Woman" if gender_predict[0] == 0 else "Man"

        # 이미지에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 1)

        # 이미지에 나이와 성별 정보 표시
        cv2.putText(frame, f"Age: {age_predict[0][0]:.1f}, Gender: {gender_str}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, w * 0.005, (255, 255, 0), 1)

    # 이미지 출력
    cv2.imshow('Age and Gender Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 웹캠 해제 및 창 닫기
camera.release()
cv2.destroyAllWindows()
</pre>
![image](https://github.com/dlwnsgur9242/Age-and-prediction-models/assets/90494150/3e8850c4-68cb-4252-9e88-4e73a50a3378)

