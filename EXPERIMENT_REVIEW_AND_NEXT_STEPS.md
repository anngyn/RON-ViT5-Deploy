# Đánh giá kết quả và kế hoạch triển khai các bước tiếp theo

## 1. Kết luận tổng quan

Kết quả hiện tại **đủ để trình bày trong báo cáo cuối kỳ với phạm vi được xác định rõ**:

> Đánh giá độ bền của ViT5 trước nhiễu OCR tổng hợp ở mức L2 trên ReceiptVQA, kèm pilot mT5 và BARTpho để kiểm tra tính lặp lại của pattern vulnerability theo severity.

Tuy nhiên, kết quả **chưa đủ để bảo vệ các kết luận mạnh** về:

- quan hệ nhân quả giữa augmentation và mức tăng ANLS;
- ý nghĩa thống kê của các chênh lệch nhỏ;
- khả năng tổng quát rộng sang backbone khác ngoài hai pilot;
- khả năng chống chịu với lỗi của OCR engine thực tế;
- khả năng tái lập toàn bộ thí nghiệm từ workspace hiện tại.

Ba flow ViT5 là kết quả chính. Các biểu đồ mT5/BARTpho là bằng chứng pilot theo severity và noise ranking; do raw CSV/log không còn trong workspace, không dùng chúng để tuyên bố model nào tốt nhất.

## 2. Mức độ trả lời các câu hỏi nghiên cứu

| Câu hỏi nghiên cứu | Mức độ hiện tại | Nhận xét |
|---|---|---|
| Noise nào gây hại nhất? | Khá đủ tại L2 | Mixed và money đứng đầu trong một lần đánh giá |
| Noisy Aug có tốt hơn baseline? | Có bằng chứng quan sát | Avg noisy tăng 2,17 điểm, nhưng training budget gần 2 lần |
| Consistency có tốt hơn augmentation? | Đủ để mô tả thiết lập đã thử | Aug cao hơn Consistency trên cả 14 noise |
| Severity L1 đến L3 thay đổi thế nào? | Có kết quả pilot | mT5 và BARTpho đều giảm retention; raw CSV nguồn cần phục hồi để nâng mức bằng chứng |
| Kết quả có lặp lại trên backbone khác? | Có tín hiệu pilot | Noise-ranking tương quan cao giữa mT5/BARTpho; chưa đủ để tổng quát rộng |
| Kết quả có ý nghĩa thống kê? | Chưa | Không có per-sample prediction hoặc confidence interval |
| Mô hình có bền với OCR thực tế không? | Chưa đánh giá | Chỉ thêm synthetic noise vào Google OCR context có sẵn |

## 3. Tổng hợp kết quả hiện tại

| Flow | Clean | Avg noisy | Drop | Retention |
|---|---:|---:|---:|---:|
| ViT5 Baseline | 84,11 | 82,06 | 2,05 | 97,57% |
| ViT5 + Noisy Aug 2x | 85,34 | 84,23 | 1,11 | 98,69% |
| ViT5 + Consistency | 83,97 | 82,92 | 1,06 | 98,74% |

### 3.1. Kết quả pilot mT5 và BARTpho

| Backbone | Retention L1 | Retention L2 | Retention L3 | Atomic macro drop L1/L2/L3 |
|---|---:|---:|---:|---:|
| mT5 | 98,0% | 96,2% | 94,0% | 0,99 / 2,11 / 3,47 |
| BARTpho | 93,9% | 90,3% | 86,4% | 1,09 / 2,38 / 3,85 |

Noise-ranking giữa hai backbone có Spearman $\rho=0,899$ ở L1, $0,881$ ở L2 và $0,873$ ở L3. Mixed đứng đầu và money đứng thứ hai ở L3. Đây là kết quả pilot được lưu trong các biểu đồ backbone; raw CSV/log tương ứng cần được phục hồi trước khi dùng để khẳng định tổng quát.

### 3.2. Cách đọc đúng bảng kết quả

Không nên chọn phương pháp chỉ dựa trên một đại lượng `drop`:

- **Clean performance** cho biết chất lượng trên dữ liệu sạch.
- **Absolute noisy performance** cho biết chất lượng thực tế trên dữ liệu nhiễu.
- **Drop/retention** cho biết mức giữ lại so với clean của chính phương pháp đó.

Ví dụ:

- Consistency có retention cao nhất, 98,74%.
- Nhưng Noisy Aug có ANLS tuyệt đối cao hơn trên cả clean và avg noisy.
- Vì vậy, retention cao không đồng nghĩa với mô hình có accuracy tốt nhất.

### 3.2. Mức cải thiện theo từng noise

Noisy Aug cao hơn baseline trên cả 14 điều kiện đã đo. Mức tăng lớn nhất tập trung ở:

- `mixed_noise`: khoảng +6,39 điểm;
- `money_noise`: khoảng +5,10 điểm;
- `accent_removal`: khoảng +2,52 điểm;
- `character_confusion`: khoảng +2,39 điểm.

Consistency cũng tăng rõ nhất ở `mixed_noise` và `money_noise`, nhưng thấp hơn Noisy Aug trên toàn bộ 14 điều kiện trong lần đánh giá hiện tại.

## 4. Các nhận xét có thể giữ

Các nhận xét sau phù hợp với bằng chứng hiện có:

1. Trong thiết lập ViT5 tại L2, `mixed_noise` và `money_noise` gây drop lớn nhất.
2. Noisy Aug có ANLS cao hơn Consistency trên toàn bộ 14 điều kiện đã đo.
3. Noisy Aug 2x đạt clean ANLS và avg noisy ANLS cao nhất.
4. Consistency tốn compute hơn nhưng không đạt accuracy tuyệt đối bằng Noisy Aug trong cấu hình consistency hiện tại.
5. Kết quả chỉ áp dụng cho synthetic noise được thêm vào OCR context có sẵn, không phải kết quả đánh giá OCR engine thực tế.

## 5. Các nhận xét cần hạ mức khẳng định

### 5.1. Over-regularization

Nhận xét hiện tại:

> Consistency bị over-regularization.

Vấn đề: clean ANLS chỉ giảm từ 84,11 xuống 83,97, tức khoảng 0,14 điểm, và chưa có confidence interval.

Nên viết:

> Clean ANLS giảm nhẹ, phù hợp với giả thuyết over-regularization, nhưng chưa đủ bằng chứng thống kê để xác nhận nguyên nhân.

Ngoài ra, kết luận chỉ áp dụng cho consistency loss đã thử:

- global mean-pooled encoder representation;
- cosine distance;
- hệ số `beta = 0.5`;
- một seed và một training configuration.

Không nên tổng quát thành “consistency regularization nói chung kém hơn augmentation”.

### 5.2. Các noise nhẹ gần như vô hại

Nhận xét hiện tại:

> Glyph, dd, date, line shuffle và code gần như vô hại.

Nên viết:

> Các noise này có drop dưới 1 điểm trong một lần đánh giá L2; chưa thể kết luận tác động không đáng kể khi chưa có khoảng tin cậy.

### 5.3. Hiệu ứng ANLS cliff

Nhận xét “một ký tự sai làm ANLS về 0” không đúng cho mọi answer.

Ví dụ:

| Prediction | Ground truth | ANLS |
|---|---|---:|
| `125.O00` | `125.000` | 0,8571 |
| `2024` | `2O24` | 0,75 |
| `12` | `1Z` | 0,50 |
| `1` | `l` | 0 |

Nên viết:

> Hiệu ứng cliff đặc biệt nghiêm trọng với answer rất ngắn hoặc khi nhiều ký tự bị sai; một lỗi đơn lẻ trên answer dài không tự động làm ANLS về 0.

### 5.4. Money exposure giải thích chênh lệch 27 lần

Phân tích hiện tại đo pattern trên gold answer, trong khi noise thực tế được áp vào OCR context. Vì vậy, answer exposure mới là một proxy, chưa phải đo trực tiếp causal exposure.

Nên viết:

> Phân bố answer gợi ý exposure là một nguyên nhân khả dĩ; cần affected-sample analysis trên chính OCR context để kiểm chứng.

### 5.5. Augmentation gây ra mức tăng 2,17 điểm

Flow Noisy Aug cũ sử dụng khoảng:

- `N` clean samples;
- `N` noisy samples;
- tổng cộng gần `2N` samples/epoch;
- khoảng 12.972 optimizer steps/epoch với batch size 8.

Baseline chỉ có khoảng 6.486 optimizer steps/epoch. Vì vậy, chưa tách được tác động của augmentation khỏi tác động của số update lớn hơn.

Nên viết:

> Noisy Aug 2x gắn với mức tăng 2,17 điểm avg noisy, nhưng chưa thể quy toàn bộ mức tăng cho augmentation do training budget lớn hơn baseline.

## 6. Những bằng chứng còn thiếu

### Bắt buộc

- Flow 4 equal-budget chưa có kết quả.
- Không có per-sample prediction.
- Không có confidence interval.
- Benchmark noise chưa được cache theo ảnh.
- Checkpoint của ba flow ViT5 không có trong workspace.
- Phần lớn training log không được lưu.

### Cần phục hồi để nâng mức bằng chứng

- CSV nguồn cho mT5 L1/L2/L3.
- CSV nguồn cho BARTpho L1/L2/L3.
- Log và config invocation thực tế của các backbone pilot.
- Checkpoint hoặc checksum tương ứng.

Các claim cross-backbone và severity hiện được trình bày ở mức pilot, có chú thích rõ nguồn là biểu đồ tổng hợp. Nếu phục hồi được CSV, log và checkpoint, nhóm có thể nâng chúng lên mức kết luận có thể tái lập.

## 7. Kế hoạch triển khai

### P0 - Đóng băng và phục hồi artifact

**Thời gian:** khoảng 0,5 ngày  
**GPU:** không cần

#### Công việc

1. Chốt commit chứa Flow 4 và các thay đổi benchmark hiện tại.
2. Tìm lại ba checkpoint ViT5 đã train.
3. Tìm CSV/log nguồn của mT5 và BARTpho.
4. Tạo experiment manifest cho từng run, gồm:
   - Git commit;
   - config;
   - seed;
   - checkpoint hash;
   - dataset hash;
   - lệnh chạy;
   - GPU;
   - thời gian train;
   - output CSV.

#### Tiêu chí hoàn thành

- Mỗi bảng và hình trong slide truy được về một CSV nguồn.
- Không có kết quả chỉ tồn tại dưới dạng PNG.
- Các artifact không phục hồi được phải được đánh dấu rõ hoặc loại khỏi bản nộp.

### P1 - Cố định benchmark noise

**Thời gian:** 1 đến 2 ngày phát triển  
**GPU:** chỉ cần inference nếu đã có checkpoint; không cần train

#### Công việc

1. Sinh corrupted OCR một lần theo khóa:

   ```text
   image_id, noise_type, level, seed
   ```

2. Dùng cùng corrupted context cho mọi QA thuộc cùng ảnh.
3. Lưu cache và checksum.
4. Bảo đảm mọi model dùng cùng cache.
5. Xuất per-sample CSV với schema tối thiểu:

   ```text
   image_id
   question_id
   noise_type
   level
   context_changed
   prediction
   ground_truth
   anls
   model_tag
   seed
   ```

6. Chạy L1/L2/L3 trên các checkpoint hiện có.
7. Tính paired bootstrap confidence interval theo `image_id`.

Không nên bootstrap độc lập từng QA vì nhiều câu hỏi cùng dùng một receipt và không độc lập thống kê.

#### Tiêu chí hoàn thành

- Chạy lại cùng seed cho kết quả giống nhau.
- Hai model đánh giá cùng condition có cùng cache hash.
- Có paired 95% CI cho Aug-Baseline và Consistency-Baseline.
- Có thể truy từ aggregate score xuống từng prediction.

### P2 - Chạy equal-budget control

**Thời gian GPU ước lượng:** 1 đến 1,5 giờ cho hai run ViT5 1x

#### Hai run bắt buộc

1. Baseline clean 1x với seed cố định.
2. Noisy Aug 1x gồm 60% clean và 40% noisy.

#### Các yếu tố phải giữ giống nhau

- 51.886 samples/epoch;
- batch size 8;
- khoảng 6.486 optimizer steps/epoch;
- 3 epoch;
- learning rate `5e-5`;
- cùng seed;
- cùng checkpoint-selection rule;
- cùng deterministic evaluation cache.

#### Ma trận thí nghiệm cuối

| Flow | Budget | Vai trò |
|---|---:|---|
| Baseline clean | 1x | Control chính |
| Aug 60/40 | 1x | Đo tác động augmentation với budget bằng nhau |
| Aug clean + noisy | 2x | Kết quả exploratory cũ |
| Consistency | Paired | So sánh phương pháp trong thiết lập đã thử |

#### Tiêu chí hoàn thành

- Baseline 1x và Aug 1x có cùng optimizer-step budget.
- Checkpoint, log và per-sample prediction được lưu.
- Báo cáo tách rõ kết quả fair-control và exploratory 2x.

Nếu tài nguyên hạn chế, đây là thí nghiệm train cần ưu tiên nhất. Chưa nên dành GPU cho Adapter hoặc RON-NACA.

### P3 - Phân tích không cần train thêm

**Thời gian:** khoảng 1 ngày

#### Công việc

1. Tính paired delta và bootstrap CI.
2. Phân tích tỷ lệ:
   - `ANLS = 0`;
   - `0 < ANLS < 1`;
   - `ANLS = 1`.
3. Affected-sample analysis:
   - noise có thực sự thay đổi context không;
   - nếu context thay đổi thì ANLS giảm bao nhiêu;
   - tỷ lệ sample bị tác động theo từng noise.
4. Breakdown theo answer type:
   - number/phone;
   - money;
   - date;
   - text.
5. Chọn tối thiểu:
   - 5 case augmentation xử lý thành công;
   - 5 case thất bại;
   - ưu tiên money và mixed noise.

#### Tiêu chí hoàn thành

- Giả thuyết money exposure được kiểm tra trên OCR context thay vì chỉ dùng gold-answer regex.
- Hiệu ứng cliff được mô tả bằng phân bố per-sample, không chỉ bằng ví dụ suy đoán.
- Các chênh lệch nhỏ được trình bày kèm CI.

### P4 - Chốt kết luận và đồng bộ tài liệu

**Thời gian:** khoảng 0,5 đến 1 ngày

#### Bảng kết quả cuối nên có

- Clean ANLS.
- Avg noisy ANLS.
- Drop.
- Retention.
- Paired delta so với baseline.
- 95% confidence interval.
- Samples/epoch.
- Optimizer steps/epoch.
- Train time.

#### Các artifact cần đồng bộ

- `RESULTS_REPORT.md`;
- slide TeX và PDF;
- `README.md`;
- trạng thái từng flow;
- đường dẫn dataset;
- CSV nguồn của mỗi hình;
- hướng dẫn tái lập.

#### Quy tắc viết kết luận

- Phân biệt “quan sát được” với “nguyên nhân được chứng minh”.
- Không gọi synthetic-noise robustness là real-OCR robustness.
- Không tổng quát một cấu hình consistency thành consistency regularization nói chung.
- Không trình bày Flow 4 như đã hoàn thành trước khi có checkpoint và CSV.

### P5 - Demo tối thiểu nếu rubric yêu cầu triển khai

**Thời gian:** khoảng 1 đến 2 ngày

Không cần xây dựng hệ thống lớn. Một CLI hoặc Gradio app nhỏ là đủ:

```text
Question + OCR text
        -> chọn checkpoint
        -> sinh prediction
        -> hiển thị clean/noisy comparison
```

Demo nên có ba receipt mẫu:

1. OCR clean.
2. Money noise.
3. Mixed noise.

Demo cần hiển thị:

- input context;
- question;
- ground truth;
- prediction;
- ANLS;
- loại và mức noise.

## 8. Thứ tự thực hiện đề xuất

```text
P0. Phục hồi artifact và đóng băng phiên bản
                  |
                  v
P1. Cố định benchmark và xuất per-sample result
                  |
                  v
P2. Train Baseline 1x và Aug 1x equal-budget
                  |
                  v
P3. Bootstrap CI và affected-sample analysis
                  |
                  v
P4. Sửa kết luận, báo cáo, slide và README
                  |
                  v
P5. Demo tối thiểu nếu rubric yêu cầu
```

## 9. Phạm vi hoàn thành hợp lý cho đồ án

Điểm dừng phù hợp cho đồ án cuối kỳ là hoàn thành **P0 đến P4**.

P5 chỉ bắt buộc nếu môn học yêu cầu sản phẩm demo. Các hướng sau nên để optional:

- retrain toàn bộ mT5/BARTpho;
- Adapter Only;
- RON-NACA;
- multi-seed training tốn nhiều GPU.

Nếu còn tài nguyên, có thể chạy thêm training seed để đo training variance. Nếu không, cần ghi rõ rằng bootstrap theo test image chỉ đo test-set uncertainty, không đo training-seed uncertainty.

## 10. Ưu tiên ngắn gọn

| Ưu tiên | Việc cần làm | Lý do |
|---|---|---|
| 1 | Phục hồi checkpoint và CSV nguồn | Không có artifact thì không thể tái lập |
| 2 | Cố định cached benchmark | Loại bỏ phụ thuộc thứ tự RNG |
| 3 | Chạy Baseline 1x và Aug 1x | Trả lời confound quan trọng nhất |
| 4 | Per-sample output và bootstrap CI | Làm kết luận có độ tin cậy |
| 5 | Affected-sample analysis | Kiểm chứng money exposure và cliff effect |
| 6 | Đồng bộ báo cáo/slide/README | Tránh claim vượt bằng chứng |
| 7 | Demo | Chỉ khi rubric yêu cầu |
