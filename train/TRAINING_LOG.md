# 학습 로그

각 버전 학습 시 콘솔 출력을 정리한 기록. 사전학습 백본 다운로드 진행바 등 잡음은 제거했다.

## v1 — `v1_market_resnet18.py` (Market, resnet18)

```
[train] identity 751 / images 12936 / device cuda
[train] train 11643 / val 1293
  ep   1  gender 0.910  upper 0.858  lower 0.825  mean 0.865
  ep   2  gender 0.922  upper 0.881  lower 0.853  mean 0.885
  ep   3  gender 0.934  upper 0.911  lower 0.870  mean 0.905
  ep   4  gender 0.940  upper 0.927  lower 0.893  mean 0.920
  ep   5  gender 0.944  upper 0.922  lower 0.899  mean 0.921
  ep   6  gender 0.949  upper 0.921  lower 0.912  mean 0.927
  ep   7  gender 0.960  upper 0.932  lower 0.886  mean 0.926
  ep   8  gender 0.947  upper 0.930  lower 0.904  mean 0.927
  ep   9  gender 0.942  upper 0.936  lower 0.909  mean 0.929
  ep  10  gender 0.951  upper 0.926  lower 0.901  mean 0.926
  ep  11  gender 0.960  upper 0.913  lower 0.902  mean 0.925
  ep  12  gender 0.953  upper 0.922  lower 0.891  mean 0.922
  ep  13  gender 0.954  upper 0.935  lower 0.885  mean 0.924
  ep  14  gender 0.937  upper 0.930  lower 0.890  mean 0.919
  ep  15  gender 0.954  upper 0.921  lower 0.890  mean 0.922
  ep  16  gender 0.952  upper 0.927  lower 0.888  mean 0.922
  ep  17  gender 0.948  upper 0.933  lower 0.891  mean 0.924
  ep  18  gender 0.959  upper 0.944  lower 0.908  mean 0.937
  ep  19  gender 0.950  upper 0.949  lower 0.933  mean 0.944
  ep  20  gender 0.961  upper 0.943  lower 0.914  mean 0.939
[train] 완료. best mean_acc=0.944 -> weights/color_par_v1_market_resnet18.pt
```

## v2 — `v2_peta_resnet50.py` (PETA, resnet50, 11색)

```
[peta] 학습대상 이미지 18986 / device cuda
[peta] train 17381 / val 1605 (인물 8691명)
  ep   1  gender 0.844  upper 0.628  lower 0.554  mean 0.675
  ep   2  gender 0.846  upper 0.662  lower 0.631  mean 0.713
  ep   3  gender 0.857  upper 0.648  lower 0.627  mean 0.711
  ep   4  gender 0.872  upper 0.634  lower 0.612  mean 0.706
  ep   5  gender 0.855  upper 0.699  lower 0.672  mean 0.742
  ep   6  gender 0.855  upper 0.697  lower 0.692  mean 0.748
  ep   7  gender 0.864  upper 0.709  lower 0.707  mean 0.760
  ep   8  gender 0.865  upper 0.702  lower 0.717  mean 0.762
  ep   9  gender 0.860  upper 0.693  lower 0.713  mean 0.755
  ep  10  gender 0.891  upper 0.710  lower 0.723  mean 0.775
  ep  11  gender 0.874  upper 0.695  lower 0.684  mean 0.751
  ep  12  gender 0.883  upper 0.680  lower 0.697  mean 0.753
  ep  13  gender 0.878  upper 0.691  lower 0.718  mean 0.762
  ep  14  gender 0.855  upper 0.713  lower 0.717  mean 0.762
  ep  15  gender 0.898  upper 0.697  lower 0.702  mean 0.765
  ep  16  gender 0.862  upper 0.686  lower 0.704  mean 0.751
  ep  17  gender 0.881  upper 0.712  lower 0.726  mean 0.773
  ep  18  gender 0.900  upper 0.716  lower 0.703  mean 0.773
  ep  19  gender 0.887  upper 0.724  lower 0.694  mean 0.768
  ep  20  gender 0.892  upper 0.703  lower 0.735  mean 0.777
  ep  21  gender 0.850  upper 0.688  lower 0.720  mean 0.753
  ep  22  gender 0.886  upper 0.692  lower 0.708  mean 0.762
  ep  23  gender 0.880  upper 0.692  lower 0.703  mean 0.758
  ep  24  gender 0.881  upper 0.713  lower 0.698  mean 0.764
  ep  25  gender 0.885  upper 0.725  lower 0.720  mean 0.777
[peta] 완료. best mean_acc=0.777 -> weights/color_par_v2_peta_resnet50.pt
```

## v3 — `v4_multi_resnet50_sleeve.py`의 소매 헤드 추가 전 버전 (PETA+Market, resnet50, 3헤드)

```
[multi] PETA+Market 통합 이미지 31922 / backbone resnet50 / cuda
[multi] train 28558 / val 3364 (인물 9442명)
  ep   1  gender 0.830  upper 0.662  lower 0.584  mean 0.692
  ep   2  gender 0.861  upper 0.677  lower 0.691  mean 0.743
  ep   3  gender 0.844  upper 0.671  lower 0.683  mean 0.733
  ep   4  gender 0.875  upper 0.703  lower 0.682  mean 0.753
  ep   5  gender 0.863  upper 0.686  lower 0.655  mean 0.734
  ep   6  gender 0.842  upper 0.689  lower 0.682  mean 0.738
  ep   7  gender 0.868  upper 0.698  lower 0.658  mean 0.741
  ep   8  gender 0.874  upper 0.682  lower 0.699  mean 0.752
  ep   9  gender 0.858  upper 0.694  lower 0.724  mean 0.759
  ep  10  gender 0.864  upper 0.644  lower 0.647  mean 0.718
  ep  11  gender 0.860  upper 0.693  lower 0.722  mean 0.759
  ep  12  gender 0.887  upper 0.687  lower 0.719  mean 0.764
  ep  13  gender 0.882  upper 0.666  lower 0.700  mean 0.749
  ep  14  gender 0.874  upper 0.699  lower 0.710  mean 0.761
  ep  15  gender 0.880  upper 0.692  lower 0.725  mean 0.766
  ep  16  gender 0.861  upper 0.646  lower 0.716  mean 0.741
  ep  17  gender 0.856  upper 0.657  lower 0.711  mean 0.741
  ep  18  gender 0.865  upper 0.681  lower 0.689  mean 0.745
  ep  19  gender 0.865  upper 0.688  lower 0.726  mean 0.760
  ep  20  gender 0.864  upper 0.705  lower 0.691  mean 0.753
  ep  21  gender 0.875  upper 0.699  lower 0.713  mean 0.762
  ep  22  gender 0.885  upper 0.703  lower 0.717  mean 0.768
  ep  23  gender 0.864  upper 0.697  lower 0.683  mean 0.748
  ep  24  gender 0.887  upper 0.690  lower 0.711  mean 0.763
  ep  25  gender 0.871  upper 0.708  lower 0.703  mean 0.761
[multi] 완료. best mean_acc=0.768 -> weights/color_par_v3_multi_resnet50.pt
```

### 같은 v3, backbone만 swin_t로 교체 (미채택 — resnet50과 성능 차이 작고 무거움)

```
[multi] PETA+Market 통합 이미지 31922 / backbone swin_t / cuda
[multi] train 28558 / val 3364 (인물 9442명)
  ep   1  gender 0.758  upper 0.688  lower 0.704  mean 0.717
  ep   2  gender 0.858  upper 0.680  lower 0.585  mean 0.708
  ep   3  gender 0.842  upper 0.590  lower 0.551  mean 0.661
  ep   4  gender 0.825  upper 0.707  lower 0.671  mean 0.734
  ep   5  gender 0.860  upper 0.726  lower 0.691  mean 0.759
  ep   6  gender 0.874  upper 0.690  lower 0.651  mean 0.738
  ep   7  gender 0.868  upper 0.700  lower 0.725  mean 0.764
  ep   8  gender 0.865  upper 0.684  lower 0.669  mean 0.739
  ep   9  gender 0.870  upper 0.714  lower 0.732  mean 0.772
  ep  10  gender 0.856  upper 0.701  lower 0.721  mean 0.760
  ep  11  gender 0.861  upper 0.716  lower 0.675  mean 0.751
  ep  12  gender 0.867  upper 0.717  lower 0.716  mean 0.767
  ep  13  gender 0.867  upper 0.722  lower 0.717  mean 0.769
  ep  14  gender 0.859  upper 0.709  lower 0.718  mean 0.762
  ep  15  gender 0.870  upper 0.702  lower 0.712  mean 0.761
  ep  16  gender 0.864  upper 0.680  lower 0.735  mean 0.760
  ep  17  gender 0.864  upper 0.751  lower 0.744  mean 0.786
  ep  18  gender 0.869  upper 0.729  lower 0.748  mean 0.782
  ep  19  gender 0.870  upper 0.694  lower 0.719  mean 0.761
  ep  20  gender 0.862  upper 0.695  lower 0.750  mean 0.769
  ep  21  gender 0.869  upper 0.676  lower 0.697  mean 0.747
  ep  22  gender 0.864  upper 0.704  lower 0.694  mean 0.754
  ep  23  gender 0.856  upper 0.724  lower 0.712  mean 0.764
  ep  24  gender 0.862  upper 0.731  lower 0.722  mean 0.772
  ep  25  gender 0.855  upper 0.702  lower 0.715  mean 0.757
[multi] 완료. best mean_acc=0.786 -> weights/color_par_multi_swin_t.pt
```

## v4 — `v4_multi_resnet50_sleeve.py` (+ 소매 헤드)

이 실행의 콘솔 로그는 별도로 저장되지 않았다(가중치 파일만 남음 — `weights/color_par_v4_multi_resnet50_sleeve.pt`).
실전 test 15장 평가 결과는 루트 [README.md](../README.md)의 "모델 비교" 참고.

## v5 — `v5_multi_resnet50_aug.py` (v4 + 강한 증강/샘플러)

```
[v5] images 31487 / train 28128 / val 3359 / cuda
  ep   1  gen 0.838  up 0.709  low 0.692  slv 0.946  mean 0.796
  ep   2  gen 0.854  up 0.712  low 0.707  slv 0.946  mean 0.805
  ep   3  gen 0.840  up 0.681  low 0.700  slv 0.958  mean 0.795
  ep   4  gen 0.841  up 0.710  low 0.723  slv 0.957  mean 0.808
  ep   5  gen 0.858  up 0.688  low 0.701  slv 0.956  mean 0.801
  ep   6  gen 0.850  up 0.693  low 0.720  slv 0.958  mean 0.805
  ep   7  gen 0.840  up 0.702  low 0.707  slv 0.950  mean 0.800
  ep   8  gen 0.849  up 0.694  low 0.696  slv 0.959  mean 0.799
  ep   9  gen 0.870  up 0.708  low 0.698  slv 0.961  mean 0.809
  ep  10  gen 0.872  up 0.717  low 0.709  slv 0.958  mean 0.814
  ep  11  gen 0.858  up 0.690  low 0.728  slv 0.954  mean 0.808
  ep  12  gen 0.865  up 0.676  low 0.700  slv 0.950  mean 0.798
  ep  13  gen 0.865  up 0.707  low 0.702  slv 0.962  mean 0.809
  ep  14  gen 0.872  up 0.690  low 0.713  slv 0.960  mean 0.809
  ep  15  gen 0.881  up 0.714  low 0.716  slv 0.955  mean 0.816
  ep  16  gen 0.873  up 0.720  low 0.697  slv 0.954  mean 0.811
  ep  17  gen 0.879  up 0.703  low 0.715  slv 0.960  mean 0.814
  ep  18  gen 0.867  up 0.742  low 0.719  slv 0.962  mean 0.823
  ep  19  gen 0.878  up 0.730  low 0.719  slv 0.957  mean 0.821
  ep  20  gen 0.876  up 0.706  low 0.699  slv 0.960  mean 0.810
  ep  21  gen 0.863  up 0.701  low 0.726  slv 0.962  mean 0.813
  ep  22  gen 0.880  up 0.716  low 0.703  slv 0.960  mean 0.815
  ep  23  gen 0.879  up 0.714  low 0.714  slv 0.963  mean 0.817
  ep  24  gen 0.882  up 0.723  low 0.724  slv 0.960  mean 0.822
  ep  25  gen 0.877  up 0.710  low 0.724  slv 0.962  mean 0.818
  ep  26  gen 0.879  up 0.713  low 0.723  slv 0.951  mean 0.817
  ep  27  gen 0.894  up 0.735  low 0.732  slv 0.962  mean 0.831
  ep  28  gen 0.884  up 0.724  low 0.729  slv 0.961  mean 0.825
  ep  29  gen 0.886  up 0.732  low 0.737  slv 0.965  mean 0.830
  ep  30  gen 0.874  up 0.725  low 0.734  slv 0.963  mean 0.824
  ep  31  gen 0.881  up 0.732  low 0.726  slv 0.964  mean 0.826
  ep  32  gen 0.881  up 0.729  low 0.728  slv 0.965  mean 0.826
  ep  33  gen 0.891  up 0.735  low 0.733  slv 0.966  mean 0.831
  ep  34  gen 0.879  up 0.728  low 0.737  slv 0.965  mean 0.827
  ep  35  gen 0.881  up 0.729  low 0.743  slv 0.962  mean 0.829
  ep  36  gen 0.881  up 0.732  low 0.742  slv 0.965  mean 0.830
  ep  37  gen 0.882  up 0.733  low 0.740  slv 0.965  mean 0.830
  ep  38  gen 0.885  up 0.734  low 0.735  slv 0.964  mean 0.829
  ep  39  gen 0.887  up 0.737  low 0.737  slv 0.963  mean 0.831
  ep  40  gen 0.888  up 0.736  low 0.740  slv 0.963  mean 0.832
[v5] 완료. best mean_acc=0.832 -> weights/color_par_v5_multi_resnet50_aug.pt
```

> **참고** — 위 val 정확도(예: v4/v5 mean 0.83)는 학습 중 val 분할 기준이고, 루트 README의
> "모델 비교" 표(test 15장 기준)와는 다른 수치다. val은 같은 데이터셋 분포, test는
> 실제 카메라 사진이라 더 어렵다.
