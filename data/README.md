# data/

학습에 쓰는 원본 데이터셋이 들어가는 자리. 용량이 크고(500MB+, 8.7만 파일) 공개
데이터셋이라 git에는 커밋하지 않는다(`.gitignore` 참고). 아래 구조로 받아서 배치할 것.

```
data/
├── Market-1501-v15.09.15/     # http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
├── Market-1501_Attribute/     # https://github.com/vana77/Market-1501_Attribute
└── PETA dataset/              # http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
```
