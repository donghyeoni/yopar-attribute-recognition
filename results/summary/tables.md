### L-a Data split per training run

| version | log | images | train | val | persons | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | `v1_market_resnet18.log` | 12936 | 11643 | 1293 | 751 | 20 |
| v2 | `v2_peta_resnet50.log` | 18986 | 17381 | 1605 | 8691 | 25 |
| v3 | `v3_multi_resnet50.log` | 31922 | 28558 | 3364 | 9442 | 25 |
| v3 (swin_t) | `v3_multi_swin_t.log` | 31922 | 28558 | 3364 | 9442 | 25 |
| v5 | `v5_multi_resnet50_aug.log` | 31487 | 28128 | 3359 | – | 40 |

### L-b Epochs with the best printed validation mean

| version | best mean | epoch | gender | upper | lower | sleeve |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.944 | 19 | 0.950 | 0.949 | 0.933 | – |
| v2 | 0.777 | 20, 25 | 0.892 / 0.885 | 0.703 / 0.725 | 0.735 / 0.720 | – / – |
| v3 | 0.768 | 22 | 0.885 | 0.703 | 0.717 | – |
| v3 (swin_t) | 0.786 | 17 | 0.864 | 0.751 | 0.744 | – |
| v5 | 0.832 | 40 | 0.888 | 0.736 | 0.740 | 0.963 |

### L-c v4 on the validation split (3359 images)

| head | accuracy | macro-F1 | mA | top-2 accuracy |
| --- | --- | --- | --- | --- |
| gender | 0.8815 | 0.8801 | 0.8808 | – |
| upper | 0.7264 | 0.6473 | 0.8069 | 0.8604 |
| lower | 0.7210 | 0.6290 | 0.7653 | 0.8776 |
| sleeve | 0.9574 | 0.9574 | 0.9577 | – |
| mean of 4 | 0.8216 | – | – | – |
| all 4 correct | 0.4701 | – | – | – |

### L-d v4 per-class metrics, upper color (validation split)

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| black | 1015 | 0.7477 | 0.8966 | 0.8154 |
| blue | 307 | 0.6988 | 0.5668 | 0.6259 |
| brown | 125 | 0.5789 | 0.7920 | 0.6689 |
| green | 210 | 0.7672 | 0.6905 | 0.7268 |
| gray | 568 | 0.7224 | 0.4489 | 0.5537 |
| orange | 27 | 0.5938 | 0.7037 | 0.6441 |
| pink | 22 | 0.6667 | 0.2727 | 0.3871 |
| purple | 101 | 0.5692 | 0.3663 | 0.4458 |
| red | 174 | 0.6857 | 0.8276 | 0.7500 |
| white | 710 | 0.7572 | 0.8169 | 0.7859 |
| yellow | 100 | 0.7245 | 0.7100 | 0.7172 |

### L-e v4 per-class metrics, lower color (validation split)

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| black | 1513 | 0.8400 | 0.8777 | 0.8584 |
| blue | 551 | 0.7268 | 0.7241 | 0.7255 |
| brown | 250 | 0.5362 | 0.5040 | 0.5196 |
| green | 55 | 0.5273 | 0.5273 | 0.5273 |
| gray | 712 | 0.5694 | 0.5126 | 0.5395 |
| orange | 4 | – | 0.0000 | – |
| pink | 43 | 0.5625 | 0.8372 | 0.6729 |
| purple | 3 | 0.0000 | 0.0000 | – |
| red | 5 | 0.3571 | 1.0000 | 0.5263 |
| white | 204 | 0.6131 | 0.5980 | 0.6055 |
| yellow | 19 | 0.7500 | 0.6316 | 0.6857 |

### L-f Own test photos (15 images, as recorded; counts = value × 15)

| version | setup | gender | upper | lower | sleeve | heads | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Market, resnet18 | 11/15 | 8/15 | 11/15 | – | 3 | 0.6667 |
| v2 | PETA, resnet50 (11 colors) | 14/15 | 7/15 | 13/15 | – | 3 | 0.7556 |
| v3 | PETA+Market | 13/15 | 8/15 | 15/15 | – | 3 | 0.8000 |
| v4 | PETA+Market + sleeve | 13/15 | 12/15 | 13/15 | 15/15 | 4 | 0.8833 |
| v5 | v4 + strong augmentation/sampler | 12/15 | 10/15 | 13/15 | 15/15 | 4 | 0.8333 |

