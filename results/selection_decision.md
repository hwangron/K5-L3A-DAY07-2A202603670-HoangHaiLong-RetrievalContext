# Quyết định chọn cấu hình (trên dev, trước khi chạy test)

Ngày 2026-11-15. Áp tiêu chí trong `selection_criteria.md`. Lúc này test chưa chạy lần nào.

Agent v3, dev, 18 câu có đáp án và 3 câu thiếu evidence:

| Cấu hình | Grounded trung bình | Context Evidence | Abstention đúng | Từ chối nhầm | Chi phí LLM | Độ trễ trung vị |
|---|---|---|---|---|---|---|
| A0_dense_starter | 0,78 | 3/18 | 3/3 | 9/18 | $0,0325 | 5,37 s |
| A1_dense_budget | 1,78 | 14/18 | 3/3 | 2/18 | $0,0312 | 3,37 s |
| A2_hybrid_rerank | 1,89 | 16/18 | 3/3 | 0/18 | $0,0279 | 4,10 s |
| A3_hybrid_rrf | 1,94 | 16/18 | 3/3 | 0/18 | $0,0301 | 3,18 s |

1. Grounded trung bình: A3 (1,94) cao hơn A2 (1,89).

**Chọn A3_hybrid_rrf, agent v3.** Baseline để so trên test là `A0_dense_starter`, agent v1 (bản đầu tiên, tham số mặc định của starter).

Lưu ý: A3 chỉ hơn A2 một câu (q02), nên khác biệt nhỏ. A3 không cần model rerank và có độ trễ thấp hơn. Sau bước này không đổi code, prompt hay tham số.
