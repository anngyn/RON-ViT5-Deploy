# Kịch bản thuyết trình 2 người

**Người trình bày chính: Ấn.** **Người phụ trách phương pháp và demo: Điền.**

Thời lượng mục tiêu khoảng 16 phút 25 giây, trong đó demo chiếm 4 phút. Webcam của người đang nói luôn bật ở góc dưới phải, không che biểu đồ.

## Phân vai theo slide

- **Ấn:** slide 1, 4, 5, 10, 11, 12, mở đầu/kết thúc demo, slide 16, 18.
- **Điền:** slide 2, 3, 6, 7, 8, 9, 13, 14, 15, phần thao tác demo, slide 17, 19.

## Lời dẫn ngắn

**Slide 1 — Ấn:** “Nhóm chúng em trình bày độ bền của hỏi đáp hóa đơn tiếng Việt trước lỗi OCR, gồm bài toán, thực nghiệm, kết quả L2 và demo tái lập.”

**Slide 2 — Điền:** “Đầu vào là câu hỏi và văn bản OCR, đầu ra là đáp án ngắn như tổng tiền hoặc ngày. OCR sai có thể làm đáp án sai dù câu hỏi không đổi.”

**Slide 3 — Điền:** “Hóa đơn sạch có tổng tiền 125.000 đồng; OCR nhiễu thành 125 OOO d. Sai một ký tự ở trường tiền có thể đổi đáp án.”

**Slide 4 — Ấn:** “Mục tiêu là baseline ViT5, 14 loại nhiễu L2 và so sánh Noisy Aug với Consistency. Phạm vi là một backbone, một seed và synthetic noise.”

**Slide 5 — Ấn:** “ReceiptVQA có 9.500 hóa đơn OCR, 51.886 câu hỏi train và 6.500 test; nhóm dùng split gốc.”

**Slide 6–9 — Điền:** “Pipeline đi từ OCR sạch sang sinh nhiễu, train ba phương pháp và đánh giá trên cùng test set. 14 nhiễu gồm sai ký tự, mất thông tin, cấu trúc và trường quan trọng. Mô hình là ViT5-base, full data, 3 epoch, ANLS tại L2.”

**Slide 10–12 — Ấn:** “Noisy Aug đạt 85.34 clean, 84.23 noisy và drop 1.11 trong lần chạy này. Mixed noise drop 9.88, money noise drop 4.99. Gần 49% đáp án là số/số điện thoại, nên lỗi 0/O đặc biệt đáng chú ý. Noisy Aug có ngân sách dữ liệu/updates gần gấp đôi, nên chưa quy toàn bộ tăng điểm cho augmentation.”

**Slide 13–15 — Điền:** “Money và date cùng p = 0.30 nhưng drop khác 27 lần. Đây là proxy, cần affected-sample analysis. Noisy Aug cao hơn Consistency ở 14 điều kiện L2; Consistency clean thấp hơn baseline 0.14, phù hợp với giả thuyết over-regularization nhưng chưa đủ chứng minh.”

## Demo tái lập kết quả — 4 phút

**Ấn mở đầu:** “Checkout không còn checkpoint nên nhóm không demo inference từng hóa đơn. Demo này truy vết cấu hình, CSV và biểu đồ đã lưu.”

**Điền thao tác trên `demo_tai_lap_ket_qua.html`:**

1. Chọn **ViT5 baseline**, chỉ ra cấu hình clean và các chỉ số L2.
2. Chọn **ViT5 + Noisy Aug**, chỉ ra `use_noisy_aug`, tỷ lệ 1:1, clean 85.34 và noisy 84.23.
3. Chuyển **Mức ảnh hưởng nhiễu** để chỉ ra mixed/money noise.
4. Chuyển **Khả năng phục hồi** để đối chiếu Noisy Aug và Consistency.
5. Mở Terminal tại thư mục project và chạy `python scripts/plot_report_figures.py` chỉ khi đã chạy thử thành công; lệnh này sinh biểu đồ từ CSV, không cần checkpoint.

**Ấn kết thúc:** “Demo xác nhận đường đi cấu hình → CSV → biểu đồ. Đây là demo tái lập phân tích, không phải inference trực tiếp.”

**Slide 16–18 — Ấn:** “Ưu tiên tiếp theo là equal-budget run, prediction từng mẫu, bootstrap confidence interval và phân tích money/mixed. Adapter Only và RON-NACA mới là hướng mở rộng, chưa chạy. Đóng góp hiện tại là benchmark ViT5 và 14 nhiễu OCR có kiểm soát.”

**Slide 19 — Điền:** “Nhóm em xin cảm ơn thầy cô và xin nhận câu hỏi.”

## Trả lời nhanh

- **Vì sao không inference?** Checkpoint không còn; nhóm không giả lập, chỉ demo cấu hình, CSV và biểu đồ đã lưu.
- **Một seed có đủ không?** Chưa đủ cho kết luận thống kê; cần bootstrap confidence interval.
- **Noisy Aug tốt hơn do đâu?** Chưa tách được tác động augmentation khỏi số updates; cần equal-budget run.
