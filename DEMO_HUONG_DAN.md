# Demo tái lập kết quả

Mở `demo_tai_lap_ket_qua.html` bằng Chrome/Safari và bật toàn màn hình khi quay.

Kịch bản 4--5 phút:

1. Nêu rõ phạm vi: đây là demo tái lập kết quả, không chạy inference vì không còn checkpoint.
2. Chọn lần lượt Baseline, Noisy Aug và Consistency ở thanh phương pháp; giải thích các cấu hình và chỉ số tổng hợp ở L2.
3. Cuộn bảng để chỉ ra `mixed_noise` và `money_noise` có drop nổi bật.
4. Chuyển ba biểu đồ để đối chiếu kết quả tổng hợp, ranking tác động nhiễu và khả năng phục hồi.
5. Nhấn “Hiển thị bước tái lập”, rồi có thể mở terminal và chạy lệnh bên dưới nếu cần minh họa thao tác thật:

```bash
python scripts/plot_report_figures.py
```

Lệnh ghi lại ảnh biểu đồ từ CSV trong `outputs/results/`; nên chạy trước buổi quay để xác nhận môi trường có đủ thư viện.
