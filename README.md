# 타이타닉 데이터 분석 및 머신러닝 구축 과제

## 프로젝트 정보
- **기계학습 및 실습 2024**
- **교수님**: 강경수 교수님
- **학부**: 인공지능 융합학부
- **이름**: 송주빈
- **학번**: 2023100978
- **주제**: 타이타닉 데이터 분석 및 머신러닝 구축

## 1. 프로젝트 개요
Titanic 데이터셋을 사용하여 승객이 **생존했는지 여부**를 예측하는 머신러닝 모델을 개발하는 프로젝트입니다. Logistic Regression, SVM, Random Forest와 같은 분류 모델을 활용하여 성능을 비교하고 최적의 모델을 선택합니다.

## 2. 분석 목적
Titanic 생존 예측 모델을 통해 다음과 같은 목적을 달성하고자 합니다.
- **다양한 Feature**(승객의 클래스, 성별, 나이 등)와 생존 간의 관계를 분석합니다.
- 머신러닝 분류 모델을 통해 생존 여부를 예측합니다.
- 다양한 모델을 평가하고 **AUC Score**를 기준으로 최적의 모델을 선정합니다.

## 3. 구현 내용

### 3.1 데이터 전처리
- **결측치 처리**: `Age`, `Embarked`, `Fare` 컬럼의 결측치를 평균 또는 최빈값으로 대체했습니다.
- **One-Hot Encoding**: 범주형 변수(`Sex`, `Embarked`)에 대해 원-핫 인코딩을 적용했습니다.
- **특정 Feature 생성**: 이름에 포함된 호칭 정보를 추출하여 `Name_has_title` 플래그를 생성했습니다.
- **데이터 정규화**: `Age`와 `Fare` 컬럼은 `StandardScaler`를 사용해 정규화했습니다.

### 3.2 모델 학습 및 평가
- **Logistic Regression**, **SVM**, **Random Forest** 모델을 사용해 학습을 진행했습니다.
- **AUC Score**와 **ROC Curve**를 기준으로 모델 성능을 평가했습니다.
- 성능이 가장 우수했던 Logistic Regression 모델을 최종 모델로 선정했습니다.

### 3.3 모델 저장 및 예측
- **Logistic Regression 모델**을 `pickle`로 저장하여 이후 예측에 재사용할 수 있도록 했습니다.
- 테스트 데이터에 대해 생존 여부를 예측하고, 결과를 `submission.csv`로 저장했습니다.

## 4. 개발 환경
- Python 3.x
- 주요 라이브러리: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pickle`

## 5. 실행 방법

### 5.1 환경 설정
1. Python 환경 설정
   ```bash
   pip install pandas numpy scikit-learn matplotlib
# 전처리 및 학습 데이터 불러오기
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

# 전처리 함수 실행
df_train_preprocessed, replace_embarked, replace_age, scaler = part5_preprocessing(df_train, mode='train')
df_test_preprocessed, _, _, _ = part5_preprocessing(df_test, mode='test', replace_embarked=replace_embarked, replace_age=replace_age, scaler=scaler)

# Feature(X), Label(y) 분리
train_y = df_train_preprocessed['Survived']
train_X = df_train_preprocessed.drop(columns=['Survived'])
test_X = df_test_preprocessed.set_index('PassengerId')
# 모델 학습
model1 = LogisticRegression(C=20, max_iter=1000, random_state=42)
model2 = SVC(kernel='linear', probability=True, random_state=42)
model3 = RandomForestClassifier(max_depth=3, random_state=42)

model1.fit(train_X, train_y)
model2.fit(train_X, train_y)
model3.fit(train_X, train_y)

# AUC Score 계산 및 비교
models = [model1, model2, model3]
model_names = ['Logistic Regression', 'SVM', 'Random Forest']
auc_scores = compare_auc_scores(models, model_names, test_X, train_y)
plot_auc_scores_adjusted(auc_scores)
# 모델 학습
model1 = LogisticRegression(C=20, max_iter=1000, random_state=42)
model2 = SVC(kernel='linear', probability=True, random_state=42)
model3 = RandomForestClassifier(max_depth=3, random_state=42)

model1.fit(train_X, train_y)
model2.fit(train_X, train_y)
model3.fit(train_X, train_y)

# AUC Score 계산 및 비교
models = [model1, model2, model3]
model_names = ['Logistic Regression', 'SVM', 'Random Forest']
auc_scores = compare_auc_scores(models, model_names, test_X, train_y)
plot_auc_scores_adjusted(auc_scores)
# 테스트 데이터 예측
pred_y = model1.predict(test_X)
df_result = pd.DataFrame({'PassengerId': test_X.index, 'Survived': pred_y})

# CSV 파일로 저장
df_result.to_csv('submission.csv', index=False)

pip install pandas numpy scikit-learn matplotlib
python <script_name.py>
