# Kịch bản thuyết trình 2 người — bản nói tự nhiên

**Phân vai:** Ấn nói liền mạch slide 1–12. Điền nhận từ slide 13 đến slide 19, bao gồm toàn bộ demo. Chỉ chuyển người một lần sau slide 12.

**Thời lượng:** khoảng 16–18 phút, trong đó demo khoảng 4 phút. Người đang nói bật webcam ở góc phải dưới; khi chuyển người chỉ cần nói câu bàn giao, không phải dừng video.

## Chuẩn bị trước khi quay

1. Mở slide ở chế độ trình chiếu, bắt đầu từ slide 1.
2. Mở sẵn `demo_tai_lap_ket_qua.html` trong trình duyệt nhưng chưa chuyển sang màn hình demo.
3. Trong OBS, dùng một cảnh gồm slide/demo và webcam. Kiểm tra webcam nằm trên `Display Capture`.
4. Nếu không có checkpoint, không chạy inference trực tiếp. Chỉ trình bày đường đi **cấu hình → CSV → biểu đồ**.
5. Trước khi quay thật, ghi thử 30 giây để kiểm tra âm lượng, mặt và chữ trên slide.

---

## Phần 1 — Ấn nói slide 1–12

### Slide 1 — Giới thiệu — khoảng 30 giây

“Em xin chào thầy cô. Em là Ấn, hôm nay nhóm em gồm Ấn và Điền sẽ trình bày báo cáo cuối kỳ về độ bền của mô hình hỏi đáp hóa đơn tiếng Việt trước lỗi OCR.

Nói đơn giản, nhóm đưa vào một câu hỏi cùng phần văn bản OCR của hóa đơn, sau đó mô hình phải tìm ra câu trả lời ngắn như tổng tiền, ngày hoặc tên cửa hàng. Trong bài này, nhóm tập trung xem khi OCR bị sai thì mô hình giảm chất lượng như thế nào, và cách huấn luyện với dữ liệu nhiễu có giúp mô hình bền hơn không.

Phần trình bày gồm bốn nội dung: bài toán và dữ liệu, cách tạo nhiễu và huấn luyện, kết quả thực nghiệm, cuối cùng là demo cách truy vết các kết quả đã lưu.”

### Slide 2 — Bài toán — khoảng 45 giây

“Trước hết là bài toán. Đầu vào có hai phần: câu hỏi tiếng Việt và văn bản OCR được đọc từ hóa đơn. Đầu ra là một đáp án ngắn.

Vấn đề nằm ở chỗ câu hỏi có thể không đổi nhưng văn bản OCR lại thay đổi. Ví dụ số 0 bị đọc thành chữ O, dấu chấm bị mất hoặc một dòng bị cắt. Khi đó mô hình có thể không tìm thấy đúng thông tin dù thông tin thật vẫn có trên hóa đơn.

Vì vậy nhóm không chỉ đo điểm trên OCR sạch. Nhóm còn tạo các phiên bản OCR bị nhiễu, rồi đo mức giảm ANLS giữa dữ liệu sạch và dữ liệu nhiễu. Mục tiêu là biết loại lỗi nào gây hại nhiều nhất và phương pháp nào giảm được mức giảm đó.”

### Slide 3 — Ví dụ trực quan — khoảng 40 giây

“Ở slide này là một ví dụ cụ thể. Với OCR sạch, trường tổng tiền có thể đọc là 125.000 đồng. Khi thêm nhiễu, chuỗi có thể biến thành dạng ‘125 OOO d’.

Với người nhìn, chúng ta vẫn đoán được đây là một con số. Nhưng với bài toán hỏi đáp, chỉ cần sai một ký tự ở trường tiền là đáp án sinh ra có thể không còn khớp. Đây là lý do nhóm tách riêng các lỗi liên quan đến tiền, ngày và số.”

### Slide 4 — Mục tiêu — khoảng 45 giây

“Từ vấn đề đó, nhóm đặt ra ba mục tiêu.

Thứ nhất, xây một baseline ViT5 trên dữ liệu OCR sạch. Thứ hai, tạo 14 loại nhiễu có kiểm soát ở mức L2 để biết từng loại nhiễu ảnh hưởng ra sao. Thứ ba, so sánh hai hướng huấn luyện là Noisy Aug và Consistency.

Ba câu hỏi nhóm trả lời là: nhiễu nào gây giảm điểm mạnh nhất, thêm dữ liệu nhiễu có giúp mô hình phục hồi không, và consistency có giữ được điểm trên dữ liệu sạch hay phải đánh đổi. Phạm vi kết luận của báo cáo là một backbone ViT5, một seed và synthetic noise.”

### Slide 5 — Dữ liệu — khoảng 45 giây

“Dữ liệu sử dụng là ReceiptVQA. Toàn bộ tập có khoảng 9.500 hóa đơn OCR, khoảng 51.886 câu hỏi trong train và khoảng 6.500 câu hỏi trong test.

Nhóm giữ split gốc để các phương pháp được so sánh trên cùng một nền. Phần ảnh bên phải là ví dụ hóa đơn, còn mô hình thực tế nhận phần OCR đã trích xuất cùng câu hỏi.

Điểm quan trọng ở đây là nhóm không thay đổi test set giữa các phương pháp. Nhờ vậy, chênh lệch điểm ở các slide sau đến từ cách huấn luyện và loại nhiễu, không phải do đổi bộ kiểm tra.”

### Slide 6 — Pipeline — khoảng 50 giây

“Pipeline được thực hiện theo thứ tự như sau.

Đầu tiên, nhóm dùng OCR sạch để huấn luyện baseline. Tiếp theo, từ cùng dữ liệu sạch đó, nhóm sinh các phiên bản có nhiễu. Các phiên bản này được dùng để huấn luyện Noisy Aug và Consistency.

Cuối cùng, cả ba phương pháp đều được đánh giá trên cùng test set ở hai trạng thái: clean và noisy. Kết quả được lưu thành CSV theo từng loại nhiễu, sau đó mới tổng hợp thành các biểu đồ trong báo cáo.

Như vậy, đường đi của dữ liệu là: ReceiptVQA, sinh nhiễu, huấn luyện ba flow, đánh giá L2 và lưu kết quả.”

### Slide 7 — 14 loại nhiễu — khoảng 45 giây

“Mười bốn loại nhiễu được chia thành bốn nhóm để dễ theo dõi.

Nhóm thứ nhất là sai ký tự, ví dụ nhầm 0 và O. Nhóm thứ hai là mất thông tin, chẳng hạn xóa một phần token. Nhóm thứ ba là nhiễu cấu trúc như đổi khoảng trắng hoặc thứ tự. Nhóm cuối cùng là nhiễu vào trường quan trọng như tiền, ngày và mã.

Ngoài từng loại riêng lẻ, nhóm còn có mixed noise, tức là kết hợp nhiều lỗi trong cùng một mẫu để mô phỏng tình huống OCR xấu hơn.”

### Slide 8 — Thiết lập — khoảng 45 giây

“Về thiết lập, backbone là ViT5-base. Input được giới hạn 256 token. Chỉ số chính là ANLS với ngưỡng khớp 0,5.

Các flow được train trên full data trong ba epoch. Noise generator dùng seed 42 và tất cả phương pháp dùng cùng test set. Những điểm này giúp kết quả có thể đối chiếu trực tiếp, dù chúng ta vẫn phải nhớ rằng hiện tại mới có một seed.”

### Slide 9 — Phạm vi đánh giá và đầu ra — khoảng 35 giây

“Trong phạm vi báo cáo cuối kỳ, nhóm đã chạy đủ ba phương pháp và 14 điều kiện nhiễu ở L2. Các đầu ra dùng cho phân tích gồm CSV đánh giá, bảng ranking mức ảnh hưởng của noise và biểu đồ recovery.

Các giới hạn cần ghi rõ là chưa có equal-budget, chưa có nhiều seed và chưa lưu prediction theo từng mẫu. Vì vậy nhóm sẽ phân biệt rõ kết quả quan sát được trong cấu hình này với những giả thuyết chưa thể khẳng định rộng hơn.”

### Slide 10 — Kết quả tổng hợp — khoảng 1 phút 20 giây

“Đây là bảng kết quả tổng hợp.

Trong lần chạy hiện tại, Noisy Aug đạt ANLS 85,34 trên clean và 84,23 trung bình trên noisy, tức mức giảm là 1,11 điểm. Baseline có mức giảm khoảng 2,05 điểm. Consistency có mức giảm gần Noisy Aug, nhưng điểm clean thấp hơn baseline.

Có một lưu ý rất quan trọng khi đọc con số này: Noisy Aug dùng cả N mẫu sạch và N mẫu nhiễu. Vì vậy số mẫu và số optimizer update gần gấp đôi baseline. Nói cách khác, chúng ta có thể kết luận Noisy Aug đạt điểm tuyệt đối cao nhất trong cấu hình đã chạy, nhưng chưa thể nói toàn bộ mức tăng chỉ đến từ augmentation.

Để mở rộng kết luận sau báo cáo, cần chạy equal-budget, tức giữ ngân sách update tương đương giữa các phương pháp.”

### Slide 11 — Ranking nhiễu — khoảng 50 giây

“Nếu nhìn riêng baseline, mixed noise gây giảm mạnh nhất, khoảng 9,88 điểm ANLS. Money noise đứng thứ hai, giảm khoảng 4,99 điểm. Các loại còn lại thấp hơn trong lần chạy này.

Kết quả này cho thấy lỗi kết hợp và lỗi ở trường tiền đáng được ưu tiên phân tích. Tuy nhiên, vì mới có một seed, các chênh lệch nhỏ không nên được gọi là khác biệt có ý nghĩa thống kê. Nhóm chỉ dùng biểu đồ này để chọn hướng phân tích tiếp theo.”

### Slide 12 — Phân bố đáp án — khoảng 45 giây

“Gần 49 phần trăm đáp án trong thống kê hiện tại là số hoặc số điện thoại. Đây là một lý do hợp lý khiến lỗi 0/O hoặc mất dấu phân cách có thể làm điểm ANLS giảm nhanh: đáp án ngắn nên chỉ cần sai một ký tự là đã không còn khớp.

Đây mới là phân tích ở mức phân bố, chưa phải bằng chứng nhân quả trên từng mẫu. Tiếp theo, Điền sẽ kiểm tra kỹ hơn sự khác nhau giữa money và date, sau đó demo đường đi từ cấu hình đến CSV và biểu đồ.”

---

## Phần 2 — Điền nói slide 13–19 và toàn bộ demo

### Slide 13 — Money và date — khoảng 55 giây

“Em tiếp tục từ phần phân bố đáp án. Ở đây nhóm so sánh money noise và date noise. Hai loại này dùng cùng xác suất nhiễu 0,30, nhưng mức giảm điểm rất khác nhau: money giảm khoảng 4,99, còn date chỉ giảm khoảng 0,19, chênh gần 27 lần.

Một proxy giải thích là tỷ lệ đáp án liên quan đến tiền hoặc số khoảng 48,75 phần trăm, trong khi tỷ lệ đáp án dạng ngày chỉ khoảng 5,22 phần trăm. Tuy nhiên, đây chỉ là gợi ý. Noise tác động vào toàn bộ OCR context nên chưa thể nói tỷ lệ này là nguyên nhân trực tiếp.

Một hướng kiểm chứng sau báo cáo là lưu prediction từng mẫu và kiểm tra đúng những mẫu có trường tiền bị thay đổi.”

### Slide 14 — Recovery — khoảng 40 giây

“Biểu đồ này nhìn theo hướng ngược lại: thay vì chỉ xem mô hình mất bao nhiêu điểm, chúng ta xem phương pháp nào phục hồi được bao nhiêu so với baseline.

Trong cả 14 điều kiện L2 ở lần chạy hiện tại, Noisy Aug có ANLS cao hơn Consistency. Tuy nhiên, khi đọc kết quả vẫn phải nhớ khác biệt ngân sách train giữa baseline và Noisy Aug. Vì vậy đây là kết quả mô tả của cấu hình hiện tại, chưa phải kết luận cuối cùng về phương pháp.”

### Slide 15 — Giải thích hai phương pháp — khoảng 50 giây

“Noisy Aug xem bản clean và bản noisy như các mẫu huấn luyện bổ sung, rồi tối ưu cross-entropy như bình thường.

Consistency thì thêm một ràng buộc: biểu diễn của hai phiên bản clean và noisy nên gần nhau. Ý tưởng là mô hình không nên thay đổi quá nhiều chỉ vì OCR bị nhiễu.

Trong kết quả hiện tại, Consistency có điểm clean thấp hơn baseline khoảng 0,14 điểm. Điều này phù hợp với giả thuyết over-regularization, tức ràng buộc quá mạnh làm mô hình mất một phần năng lực trên dữ liệu sạch. Nhưng vì chưa có ablation và nhiều seed, nhóm chỉ trình bày đây là giả thuyết cần kiểm tra.”

---

## Demo tái lập kết quả — khoảng 4 phút

### Bước 1 — Nói rõ phạm vi demo

“Bây giờ em chuyển sang phần demo. Trong checkout hiện tại không còn checkpoint, nên nhóm không giả lập việc chạy inference trên một hóa đơn mới. Thay vào đó, demo sẽ kiểm tra đường đi của các kết quả đã có: từ cấu hình, sang CSV, rồi đến biểu đồ báo cáo.

Cách này giúp chúng ta kiểm tra được con số trên slide lấy từ đâu mà không nói quá khả năng của artifact hiện tại.”

### Bước 2 — Mở trang demo

“Em chuyển từ slide sang trang `demo_tai_lap_ket_qua.html`. Trên trang có ba phần: chọn phương pháp, xem bảng chỉ số L2 và chuyển giữa các biểu đồ.

Đầu tiên em chọn **ViT5 baseline**. Ở phần cấu hình, baseline dùng dữ liệu clean và không bật `use_noisy_aug`. Bảng bên cạnh liệt kê kết quả theo từng noise type. Các cột clean, noisy trung bình và drop là những giá trị đã tổng hợp từ CSV.”

### Bước 3 — Đối chiếu Noisy Aug

“Tiếp theo em chọn **ViT5 + Noisy Aug**. Ở đây có thể thấy cờ `use_noisy_aug` được bật và tỷ lệ clean:noisy là 1:1.

Các con số tổng hợp hiện ra là clean 85,34 và noisy trung bình 84,23. Đây chính là hai con số được dùng trong slide kết quả tổng hợp. Em không chạy lại mô hình ở bước này; trang chỉ đọc và trình bày artifact đã lưu.”

### Bước 4 — Kiểm tra ranking và recovery

“Bây giờ em chuyển sang mục **Mức ảnh hưởng nhiễu**. Biểu đồ này cho thấy mixed noise và money noise nằm ở nhóm gây drop nổi bật, khớp với slide 11.

Sau đó em chuyển sang **Khả năng phục hồi**. Ở đây chúng ta đối chiếu Noisy Aug và Consistency theo từng loại nhiễu, khớp với slide 14.

Như vậy, demo đã đi đủ ba bước: chọn cấu hình, đọc CSV và xem biểu đồ.”

### Bước 5 — Lệnh tái lập biểu đồ

“Nếu muốn sinh lại các hình từ CSV, project có lệnh:

`python scripts/plot_report_figures.py`

Lệnh này chỉ tổng hợp kết quả và tạo biểu đồ, không cần checkpoint. Khi quay thật, em chỉ chạy lệnh nếu đã kiểm tra trước; nếu chưa, em chỉ vào lệnh và không nhấn Enter để tránh làm gián đoạn video.”

### Kết thúc demo

“Tóm lại, phần demo xác nhận đường đi **cấu hình → CSV → biểu đồ**. Đây là demo tái lập phân tích kết quả, không phải inference trực tiếp. Giới hạn này được nói rõ vì checkout hiện tại không có checkpoint.”

---

## Slide 16 — Giới hạn và hướng phát triển — khoảng 45 giây

“Để diễn giải đúng kết quả cuối kỳ, nhóm ghi nhận ba hướng phát triển.

Thứ nhất là chạy equal-budget giữa baseline và Noisy Aug để tách tác động của augmentation khỏi số optimizer update. Thứ hai là lưu prediction từng mẫu để tính bootstrap confidence interval. Thứ ba là phân tích trực tiếp các OCR context bị ảnh hưởng bởi money và mixed noise.

Kết luận hiện tại chỉ áp dụng cho synthetic noise, một seed và benchmark L2; nhóm chưa đánh giá trên OCR engine thật.”

## Slide 17 — Mở rộng sau báo cáo — khoảng 30 giây

“Sau khi benchmark công bằng hoàn tất, nhóm có thể mở rộng sang Adapter Only và RON-NACA. Hai hướng này nằm ngoài phạm vi báo cáo cuối kỳ, nên nhóm không dùng chúng để suy luận hay so sánh với các kết quả đã trình bày.”

## Slide 18 — Kết luận — khoảng 50 giây

“Tóm lại, nhóm đã xây benchmark ViT5 trên ReceiptVQA, tạo 14 loại nhiễu OCR có kiểm soát và hoàn thành đánh giá L2 cho ba flow.

Trong cấu hình đã chạy, Noisy Aug có ANLS tuyệt đối cao nhất và mức giảm thấp hơn baseline. Mixed noise và money noise là hai nhóm cần ưu tiên phân tích. Tuy nhiên, do khác biệt ngân sách train và mới có một seed, các kết quả này vẫn cần được xác nhận bằng equal-budget run và nhiều lần chạy.

Đóng góp chính của nhóm ở giai đoạn này là một quy trình đánh giá độ bền OCR có thể truy vết từ cấu hình đến CSV và biểu đồ.”

## Slide 19 — Cảm ơn — khoảng 15 giây

“Phần trình bày của nhóm em đến đây là kết thúc. Nhóm em xin cảm ơn thầy cô đã theo dõi và xin nhận câu hỏi, góp ý.”

---

## Câu trả lời ngắn khi được hỏi

- **Vì sao không demo inference?** “Checkout hiện tại không còn checkpoint. Nhóm không giả lập inference; demo chỉ truy vết cấu hình, CSV và biểu đồ đã lưu.”
- **Một seed có đủ mạnh không?** “Chưa đủ cho kết luận thống kê. Bước tiếp theo là prediction từng mẫu và bootstrap confidence interval.”
- **Noisy Aug tốt hơn do augmentation hay do nhiều update?** “Hiện chưa tách được hai yếu tố. Equal-budget run sẽ trả lời câu hỏi này.”
- **Vì sao money ảnh hưởng mạnh hơn date?** “Tỷ lệ đáp án số cao hơn là một proxy hợp lý, nhưng chưa phải quan hệ nhân quả. Cần phân tích các mẫu và OCR context bị ảnh hưởng trực tiếp.”
