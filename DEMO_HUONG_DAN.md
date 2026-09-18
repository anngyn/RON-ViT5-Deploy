# Demo tái lập kết quả

Mở `demo_tai_lap_ket_qua.html` bằng Chrome/Safari và bật toàn màn hình khi quay.

Kịch bản 4--5 phút:

1. Nêu rõ phạm vi: đây là demo tái lập kết quả, không chạy inference vì không còn checkpoint.
2. Chọn lần lượt Baseline, Noisy Aug và Consistency ở thanh phương pháp; giải thích các cấu hình và chỉ số tổng hợp ở L2.
3. Cuộn bảng để chỉ ra `mixed_noise` và `money_noise` có drop nổi bật.
4. Chuyển các biểu đồ để đối chiếu kết quả tổng hợp, ranking tác động nhiễu và khả năng phục hồi. Hai biểu đồ bổ sung **mT5/BARTpho theo severity** và **noise ranking giữa backbone** dùng để nói qua pilot cross-backbone; không gọi đây là benchmark chọn model thắng.
5. Nhấn “Hiển thị bước tái lập”, rồi có thể mở terminal và chạy lệnh bên dưới nếu cần minh họa thao tác thật:

```bash
python scripts/plot_report_figures.py
```

Lệnh ghi lại ảnh biểu đồ từ CSV trong `outputs/results/`; nên chạy trước buổi quay để xác nhận môi trường có đủ thư viện.

## Cách nói phần mT5/BARTpho

Chọn hai biểu đồ backbone trong trang demo và nói ngắn gọn: mT5 giữ retention 98,0 → 96,2 → 94,0%, BARTpho 93,9 → 90,3 → 86,4% từ L1 đến L3; mixed và money vẫn là hai nhóm gây drop nổi bật. Đây là pilot được lưu dưới dạng biểu đồ tổng hợp, còn raw CSV/log/checkpoint của hai backbone chưa có trong checkout nên không suy luận model nào tốt nhất.
