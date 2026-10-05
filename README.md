# Dacon 전기차 가격 예측 해커톤 Private 74위 / 1,314명 참여 (상위 5.6%)

전기차의 다양한 특성 데이터를 활용해 **차량 가격을 예측하는 회귀 프로젝트**입니다.

 [Dacon 대회 페이지](https://dacon.io/competitions/official/236424/overview/description)

## 프로젝트 목표

전기차의 주요 특성과 가격 간 관계를 분석하고, 머신러닝 모델을 활용해 전기차 가격을 예측하는 것을 목표로 했습니다.

## 분석 과정

### 1. 데이터 전처리
- 결측치 및 데이터 타입 확인
- 범주형 변수 `One-Hot Encoding`
- 수치형 변수 `StandardScaler`를 활용한 표준화

### 2. 모델링
- `Gradient Boosting` 기반 회귀 모델 학습
- 학습 데이터와 검증 데이터를 분리해 모델 성능 비교

### 3. 평가
모델 성능 평가는 **RMSE (Root Mean Squared Error)** 를 기준으로 진행했습니다.

## 사용 기술

`Python` `Pandas` `NumPy`  
`Scikit-learn` `Gradient Boosting`

## 결과

Dacon 전기차 가격 예측 대회에 참여해 모델을 개선하며 예측 성능을 높였고, **최종 상위 5.6%**를 기록했습니다.
