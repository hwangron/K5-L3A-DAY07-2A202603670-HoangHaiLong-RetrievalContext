# Tiêu chí chọn cấu hình

Chốt ngày 2026-11-15, trước khi đọc kết quả agent trên dev. Test chưa chạy.

Xét theo thứ tự, chỉ xét tiêu chí sau khi tiêu chí trước bằng nhau:

1. Grounded answer trung bình trên dev (thang 0–2, chấm tay).
2. Context Evidence Rate của agent trên dev.
3. Abstention đúng trên 3 câu dev thiếu evidence, và số câu có đáp án bị từ chối nhầm.
4. Chi phí và thời gian trung bình mỗi câu: rẻ hơn, nhanh hơn thì tốt hơn.

Baseline là `A0_dense_starter`. Sau khi chọn, chạy test một lần cho baseline và cấu hình chọn. Không chỉnh gì sau khi chạy test.
