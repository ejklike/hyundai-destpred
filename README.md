# hyundai-destpred

현대자동차 근미래/최종 목적지 예측 과제

## 분석환경

- Ubuntu 16.04
- Python 3.6
- Tensorflow 1.3.0

## 데이터 위치

- 모든 데이터는 `data` 폴더 내에 위치 (없으면 각자 만들 것)
- raw data는 어떠한 형식(csv, txt, tsv, ...)이든 상관 없음
  - 단, column parsing에 문제없도록 delimiter만 코드에 반영해줄 것.
  - 참고로, raw data의 컬럼 순서는 다음과 같음
    - `['car_id', 'start_dt', 'seq_id', 'x', 'y', 'link_id']`

## 진행 계획

- 9월
  - [x] data_loader
  - [ ] preprocessor
  - [ ] baseline models: feed forward network
  - [ ] initial models: recurrent neural network

- 10월
  - [ ] structure engineering
  - [ ] hyperparameter tuning
  - [ ] measure definition

- ...TBD