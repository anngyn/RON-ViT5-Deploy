# ReceiptVQA OCR Noise Robustness — Kết Quả Thí Nghiệm

Phân tích tác động của nhiễu OCR lên ViT5 cho bài toán Vietnamese ReceiptVQA, và đánh giá 2 phương pháp tăng độ bền: **Noisy Augmentation** và **Consistency Regularization**.

## 1. Thiết Lập Thí Nghiệm

| Thành phần | Giá trị |
|-----------|---------|
| Model | VietAI/vit5-base (~223M params) |
| Dataset | ReceiptVQA (train 51,886 / dev 6,426 / test 6,500) |
| Subset ratio | 1.0 (full dataset) |
| Epochs | 3 (chọn checkpoint theo dev ANLS) |
| Learning rate | 5e-5 |
| Metric | ANLS (threshold 0.5) |
| Noise level đánh giá | L2 |
| Generation | beam=4, max_length=64 |

### 3 Flows

| Flow | Method | Training data | Loss |
|------|--------|---------------|------|
| **Flow 1** | ViT5 Clean (baseline) | Clean only | CE |
| **Flow 2** | ViT5 + Noisy Aug | Clean + Noisy (2×) | CE |
| **Flow 3** | ViT5 + Consistency | Paired clean/noisy | CE_clean + CE_noisy + 0.5·Consistency |

## 2. Kết Quả Tổng Hợp (ANLS @ L2)

| Condition | Noise Type | Flow1 Clean | Flow2 Aug | Flow3 Consist |
|-----------|-----------|:-----------:|:---------:|:-------------:|
| clean | — | 0.8411 | **0.8534** | 0.8397 |
| N20 | mixed_noise | 0.7422 | **0.8062** | 0.7911 |
| N16 | money_noise | 0.7911 | **0.8421** | 0.8263 |
| N1 | accent_removal | 0.8096 | **0.8348** | 0.8248 |
| N10 | token_deletion | 0.8185 | **0.8363** | 0.8220 |
| N7 | character_deletion | 0.8204 | **0.8357** | 0.8225 |
| N5 | character_confusion | 0.8240 | **0.8480** | 0.8369 |
| N14 | token_split | 0.8270 | **0.8441** | 0.8290 |
| N2 | tone_confusion | 0.8309 | **0.8425** | 0.8296 |
| N3 | vowel_diacritic_confusion | 0.8329 | **0.8442** | 0.8357 |
| N18 | code_noise | 0.8353 | **0.8507** | 0.8370 |
| N13 | line_shuffle | 0.8359 | **0.8492** | 0.8351 |
| N17 | date_noise | 0.8392 | **0.8536** | 0.8396 |
| N6 | glyph_confusion | 0.8403 | **0.8526** | 0.8389 |
| N4 | dd_confusion | 0.8413 | **0.8519** | 0.8397 |

## 3. Phân Tích Noise Impact

### Ranking mức độ ảnh hưởng (drop from clean, baseline model)

| Rank | Noise Type | Drop | Mức độ |
|:----:|-----------|:----:|--------|
| 1 | **mixed_noise** | −9.88 | 🔴 Nghiêm trọng |
| 2 | **money_noise** | −4.99 | 🔴 Nghiêm trọng |
| 3 | accent_removal | −3.15 | 🟠 Trung bình |
| 4 | token_deletion | −2.26 | 🟠 Trung bình |
| 5 | character_deletion | −2.07 | 🟠 Trung bình |
| 6 | character_confusion | −1.70 | 🟡 Nhẹ |
| 7 | token_split | −1.40 | 🟡 Nhẹ |
| 8 | tone_confusion | −1.02 | 🟡 Nhẹ |
| 9–14 | vowel/code/line/date/glyph/dd | < −1.0 | 🟢 Không đáng kể |

### Nhận xét chính

1. **Money noise là thủ phạm đơn lẻ lớn nhất (−4.99).** Đặc thù ReceiptVQA: answer thường là số tiền; sai 1 chữ số → sai hoàn toàn (dưới ngưỡng ANLS 0.5).

2. **Mixed noise nặng nhất (−9.88)** vì kết hợp nhiều lỗi cùng lúc.

3. **Deletion > Confusion.** token_deletion (−2.26) và character_deletion (−2.07) hại hơn các loại confusion. Mất thông tin nặng hơn nhiễu thông tin.

4. **Accent removal (−3.15)** đáng chú ý — đặc thù tiếng Việt, bỏ dấu làm đổi nghĩa từ.

5. **Nhiễu nhẹ gần như vô hại (< −1.0):** glyph, dd, date, line_shuffle, code. ViT5 vốn đã bền với các nhiễu này. dd_confusion thậm chí +0.0002 (nhiễu ngẫu nhiên).

## 4. So Sánh 2 Phương Pháp

### Recovery per noise type (method − baseline)

| Noise Type | Baseline | Aug Gain | Consist Gain |
|-----------|:--------:|:--------:|:------------:|
| mixed_noise | 0.742 | **+6.4** | +4.9 |
| money_noise | 0.791 | **+5.1** | +3.5 |
| accent_removal | 0.810 | **+2.5** | +1.5 |
| character_confusion | 0.824 | **+2.4** | +1.3 |
| token_deletion | 0.819 | **+1.8** | +0.3 |
| line_shuffle | 0.836 | **+1.3** | −0.1 |
| glyph_confusion | 0.840 | **+1.2** | −0.1 |
| dd_confusion | 0.841 | **+1.1** | −0.2 |

### Kết luận so sánh

- **Noisy Aug thắng ở MỌI noise type.** aug_gain > consist_gain trên toàn bộ 14 loại.
- **Recovery tỉ lệ với severity:** noise càng hại, method càng cứu nhiều. Money & mixed được cứu nhiều nhất.
- **Consistency yếu ở noise nhẹ:** một số gain âm (glyph, line, dd, tone) → constraint làm giảm nhẹ hiệu năng.

## 5. Tại Sao Consistency Kém Hơn Noisy Aug?

Cả 2 đều huấn luyện trên clean + noisy (nội dung data giống nhau). Khác biệt nằm ở cách dùng:

| | Noisy Aug | Consistency |
|---|-----------|-------------|
| Dataset | 103,772 samples (shuffle) | 51,886 pairs |
| Loss | CE | CE_clean + CE_noisy + 0.5·(1−cos) |
| Clean ANLS | **0.8534** | 0.8397 (−1.4) |

**Nguyên nhân:**

1. **Over-regularization.** Consistency ép `h_clean ≈ h_noisy` — ràng buộc thêm làm hạn chế capacity. Đôi khi noisy input NÊN encode khác để decoder biết mà xử lý.

2. **Tín hiệu thô.** Mean-pool nén 256 tokens → 1 vector 768-chiều, mất thông tin token-level. Cosine trên vector nén là tín hiệu coarse, không dạy model xử lý noise ở token cụ thể.

3. **Bằng chứng:** Consistency clean ANLS thấp hơn baseline (0.8397 < 0.8411) — constraint làm giảm cả performance trên data sạch, dấu hiệu rõ của over-regularization.

**Bài học:** Với task này, augmentation đơn giản + CE beat consistency regularization phức tạp.

## 6. Kết Luận

1. **Không phải noise nào cũng hại** — chỉ 2–3 loại thực sự nghiêm trọng (mixed, money, accent).
2. **Money noise critical** cho ReceiptVQA (answer là số tiền).
3. **Noisy Augmentation = giải pháp tốt nhất** — đơn giản, hiệu quả, cải thiện cả clean lẫn noisy.
4. **Consistency regularization không vượt được** simple augmentation, thậm chí giảm nhẹ ở nhiễu nhẹ.

### Đề xuất triển khai
- Ưu tiên **Noisy Augmentation** với trọng tâm money + accent + deletion noise.
- Bỏ consistency loss — chi phí (paired data, 3 loss terms) không đem lại lợi ích.

## 7. Files

```
outputs/results/
├── flow1_vit_clean_noise_l2.csv      Baseline (14 noise types @ L2)
├── flow2_vit_aug_noise_l2.csv        Noisy Aug
├── flow3_consistency_noise_l2.csv    Consistency
├── flow{1,2,3}_*_anls.png            ANLS bar charts
└── flow{1,2,3}_*_drop.png            Drop-from-clean charts
```
