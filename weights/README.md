# weights/

학습된 PAR 체크포인트가 들어가는 자리. git에는 커밋하지 않고(`.gitignore` 참고 —
가장 큰 파일이 90MB대라 git 커밋 히스토리에 안 맞음)
[Releases](https://github.com/donghyeoni/YOPAR-detection/releases/tag/weights)에 올려둔다.

| 파일 | 버전 | 비고 |
|---|---|---|
| [`color_par_v1_market_resnet18.pt`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v1_market_resnet18.pt) | v1 | Market, resnet18 |
| [`color_par_v2_peta_resnet50.pt`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v2_peta_resnet50.pt) | v2 | PETA, resnet50, 11색 |
| [`color_par_v3_multi_resnet50.pt`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v3_multi_resnet50.pt) / [`.onnx`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v3_multi_resnet50.onnx) | v3 | PETA+Market 통합 |
| [`color_par_v4_multi_resnet50_sleeve.pt`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v4_multi_resnet50_sleeve.pt) / [`.onnx`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v4_multi_resnet50_sleeve.onnx) | v4 (채택) | + 소매 헤드. [yopar](https://github.com/donghyeoni/Detection_Based_on_Attribution) 실제 서비스에 배포된 버전 |
| [`color_par_v5_multi_resnet50_aug.pt`](https://github.com/donghyeoni/YOPAR-detection/releases/download/weights/color_par_v5_multi_resnet50_aug.pt) | v5 | + 강한 증강/샘플러. 미채택 — 루트 README "모델 비교" 참고 |

각 버전에 대응하는 학습 스크립트로 재생성할 수 있다(루트 README "학습" 참고).
