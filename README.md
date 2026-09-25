# YOPAR

CCTV 기반 실종자 수색 시스템에서 **실시간 AI 처리 장치(Jetson Orin Nano)** 파트다. 사람을 검출해 잘라낸 이미지에서
성별, 상의색, 하의색, 소매 길이 4속성을 예측하는 PAR(Pedestrian Attribute Recognition) 모델과, 이 모델로 카메라
4대의 영상을 실시간 분석해 서버가 등록한 인상착의와 대조하는 Jetson 서비스를 담는다. 설정·정의와 전체 결과는
[`docs/experiment-log.md`](docs/experiment-log.md)(이하 로그)에 있다.

## 전체 시스템 개요

```mermaid
flowchart LR
    CCTV["CCTV x4"] --> MEDIA["미디어 서버<br/>(Raspberry Pi 5)"]

    MEDIA -->|"실시간 영상 제공"| JETSON["실시간 AI 처리 장치<br/>(Jetson Orin Nano)"]
    MEDIA -->|"녹화본(1분 세그먼트) 업로드"| STORAGE[("영상 스토리지<br/>(S3 / MinIO)")]
    MEDIA -->|"녹화본 메타데이터 등록"| CORE["중앙 서버"]

    CORE -->|"실시간 AI 분석 작업 등록"| MQ1{{"RabbitMQ"}}
    MQ1 -->|"작업 해결"| JETSON

    JETSON -->|"실시간 발견 위치<br/>(바운딩 박스 좌표) 등록"| CORE
    JETSON -->|"탐지 이미지<br/>(crop·프레임) 업로드"| STORAGE

    CORE -->|"녹화본 AI 분석 작업 등록"| MQ2{{"RabbitMQ"}}
    MQ2 -->|"작업 해결"| WORKER["AI 분석 워커<br/>(외부 GPU 공간)"]

    WORKER -->|"분석할 녹화본 접근"| STORAGE
    WORKER -->|"녹화본 발견 위치<br/>(바운딩 박스 좌표) 등록"| CORE

    CORE -->|"신고자 화면 제공"| REPORTER["신고자 화면"]
    CORE -->|"관리자 화면 제공"| ADMIN["관리자 대시보드"]

    classDef mine fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#1f2937;
    class JETSON mine;
```

> 노란색 블록(실시간 AI 처리 장치)이 이 레포가 맡은 파트다.

| 폴더 | 내용 |
| --- | --- |
| `train/`, `eval.py` | PAR 모델 학습(v1, v2, v4, v5), 검증셋 지표, 자체 사진 평가 |
| `edge/` | Jetson 서비스: RTSP 카메라 4대 → YOLO11 사람 검출 → PAR(v4) → 인상착의 매칭 → 서버 전송 |
| `results/` | 학습 로그, v4 검증셋 지표, 자체 사진 결과, 요약 표·그림 |

## 결과

### 버전별 모델 (자체 사진 15장)

<table>
<tr>
<td>

| 버전 | 데이터 | 성별 | 상의 | 하의 | 소매 | 평균 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| v1 | Market | 11/15 | 8/15 | 11/15 | – | 0.6667 |
| v2 | PETA | 14/15 | 7/15 | 13/15 | – | 0.7556 |
| v3 | PETA+Market | 13/15 | 8/15 | 15/15 | – | 0.8000 |
| **v4** | PETA+Market | 13/15 | **12/15** | 13/15 | 15/15 | **0.8833** |
| v5 | PETA+Market | 12/15 | 10/15 | 13/15 | 15/15 | 0.8333 |

v4부터 소매 헤드가 있고, v5는 v4에 증강과<br>샘플러를 더했다. Jetson 서비스는 v4를 쓴다.

</td>
<td><img src="results/summary/test15.png" alt="그림 1" width="520"></td>
</tr>
</table>

### v4 검증셋 (3,359장)

<table>
<tr>
<td>

| 속성 | 정확도 | top-2 |
| :--- | ---: | ---: |
| 성별 | 0.8815 | – |
| 상의색 | 0.7264 | 0.8604 |
| 하의색 | 0.7210 | 0.8776 |
| 소매 | 0.9574 | – |
| 4속성 평균 | 0.8216 | – |
| 4속성 모두 정답 | 0.4701 | – |

</td>
<td><img src="results/summary/v4_confusion.png" alt="그림 2" width="620"></td>
</tr>
</table>

- 상의 gray는 568장 중 126장을 black, 104장을 white로 예측했다.

## 기타

- 학습된 가중치(v1–v5 `.pt`, v3·v4 `.onnx`)는 [Releases](https://github.com/donghyeoni/yopar-attribute-recognition/releases)에 있다.
- 버전별 학습 곡선과 클래스별 지표는 [로그](docs/experiment-log.md)에 있다.

## Contributors

- [donghyeoni](https://github.com/donghyeoni)
- [tttksj (tttksj404)](https://github.com/tttksj404)
