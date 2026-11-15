# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: OpenStax Calculus Volume 1 (2018), mục 2.3, 2.4, 3.1, 3.2; CC BY-NC-SA 4.0; 67 trang; 144 chunk (chiến lược C của Day 6, `data/day6/chunks.json`).
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16 + pgvector 0.8.7 (Docker Day 6, bảng `day6_chunks`, strategy C); `text-embedding-3-small`, 1536 chiều. Query dùng cùng model; code kiểm tra model và số chiều trước khi tìm.
- Agent model, cách chạy và phân chia dev/test: `gpt-4.1-mini`, temperature 0, tool calling. Dev: 18 câu có đáp án + 3 câu thiếu evidence. Test: 7 + 2. Lệnh chạy ở `src/README.md`.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: câu có đáp án lấy từ benchmark Day 6 (gold = nguồn, trang, trích nguyên văn nằm trọn trong một chunk). Câu thiếu evidence hỏi chủ đề gần nhưng ngoài 4 mục, đã tìm từ khóa trên 67 trang (`data/unanswerable.json`). Câu trả lời chấm 0/1/2 bằng tay theo BENCHMARK.md. Không xem test cho đến khi chốt cấu hình (commit `9599501`).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-3, filter của người dùng, 800 ký tự (A0) | Mặc định starter | dev | 9/18 | 16/18 | 5/18; agent: 3/18 | 238 ms/truy vấn |
| Phương pháp 2 | BM25, top-3, 800 ký tự | Khớp từ khóa chính xác | dev | 12/18 | 17/18 | 6/18 | 0,3 ms |
| Phương pháp 3 | Hybrid RRF (dense + BM25, 20 ứng viên, k=60), top-3, 800 ký tự | Hai cách bù lỗi nhau | dev | 12/18 | 18/18 | 6/18 | 4 ms + dense |
| Cải thiện 1 | Dense, 3000 ký tự, xếp rank-1 mọi lượt trước (A1) | 800 ký tự không chứa nổi chunk | dev | 9/18 | 16/18 | 16/18; agent: 14/18 | $0,0015/câu |
| Cải thiện 2 | Hybrid + rerank cross-encoder MiniLM-L6, 3000 ký tự (A2) | Rerank đẩy evidence lên đầu | dev | 12/18 | 17/18 | 17/18; agent: 16/18 | +771 ms (CPU) |
| Cấu hình đã chọn | Hybrid RRF, top-3, 3000 ký tự, agent v3 (A3) | Chọn trên dev | test | 4/7 | 7/7 | 7/7; agent: 6/7 | $0,0015/câu, 3,5 s |
| Baseline | Như trên, agent v1 (A0) | | test | 4/7 | 7/7 | 4/7; agent: 4/7 | $0,0012/câu, 4,6 s |

Hit@1/Hit@3 dùng câu hỏi gốc. "Agent" là context cuối sau khi agent tự viết lại query. A3 được chọn theo `results/selection_criteria.md`: grounded trung bình dev cao nhất (1,94 so với 1,89 của A2), không cần reranker, nhanh hơn.

Hai lỗi dev:
1. **Budget 800 ký tự:** chunk dài trung vị 620 ký tự nên context chứa được khoảng một chunk; A0 chỉ có 3/18 câu đủ evidence. Tăng lên 3000 ký tự: 14/18.
2. **Dense trượt từ khóa, BM25 trượt ký hiệu:** dense trượt q04, q06; BM25 trượt q02 (sách viết "M ≠ 0", câu hỏi viết "not zero"). Hybrid tìm được cả ba. Rerank lại làm mất q02 vì chấm chunk ví dụ K/0 cao hơn chunk định lý.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: hai tool `search` và `answer`; lượt đầu bắt buộc search, tối đa 3 lần search, dừng khi gọi `answer`. Chunk của mọi lượt được bỏ trùng, xếp theo hạng, đóng gói trong 3000 ký tự; model chỉ thấy context mới nhất. Citation phải là ID có trong context; một verifier (cùng model) kiểm tra từng ý với chunk được trích, sai thì sửa một lần rồi từ chối. Lỗi API/DB ghi `status: error`, không thay bằng kết quả giả (không có lỗi nào xảy ra).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A0, agent v1 / dev | 21/21 | 1: 3, 3: 18 | 9/3/6, 0,83 | 11/17 | 16/16 | 1/3 | 2/18 |
| A1, agent v3 / dev | 21/21 | 1: 15, 3: 6 | 2/0/16, 1,78 | 21/21 | 16/16 | 3/3 | 2/18 |
| A2, agent v3 / dev | 21/21 | 1: 17, 3: 4 | 1/0/17, 1,89 | 20/21 | 18/18 | 3/3 | 0/18 |
| A3, agent v3 / dev (chọn) | 21/21 | 1: 16, 2: 2, 3: 3 | 0/1/17, 1,94 | 23/23 | 18/18 | 3/3 | 0/18 |
| A0, agent v1 / test | 9/9 | 1: 4, 3: 5 | 3/0/4, 1,14 | 4/7 | 6/6 | 1/2 | 2/7 |
| A3, agent v3 / test (chọn) | 9/9 | 1: 6, 2: 1, 3: 2 | 1/0/6, 1,71 | 6/13 | 8/8 | 1/2 | 0/7 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q09 (A3, test) | Có đáp án | 1 search, p183-00 ở đầu | "Archimedes…" [p183-struct-00] | 2. Chunk nêu đúng ý |
| q09 (A0, test) | Có đáp án | 3 search, context rỗng vì chunk dài hơn 800 ký tự | Từ chối | 0. Từ chối nhầm do budget |
| u04 (A3, test) | Thiếu evidence | 3 search; verifier bác câu trả lời không có nguồn | Từ chối | Đúng. A0 (v1) trả lời product rule từ kiến thức ngoài |
| q16 (A3, test) | Có đáp án | 2 search, chỉ tìm được đề Example 2.37 (p197-01), thiếu lời giải (p198-00) | "IVT không chứng minh f có nghiệm" [p197-struct-01] | 0. Hiểu sai; Day 6 tách đề và lời giải thành hai chunk; verifier để lọt |
| u05 (A3, test) | Thiếu evidence | 3 search | Gọi `answer` với "tài liệu không nêu năm" kèm 4 citation | Sai. Không bịa năm nhưng không abstain |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: `results/retrieval/<split>/`; `results/agent/v<phiên bản>/<cấu hình>_<split>/` (`traces.json`, `labels.json`, `reviews.json`, `summary.json`); điểm context test ở `results/agent/test_context_grades.json` (baseline: 4 câu điểm 2, 3 câu điểm 0; cấu hình chọn: 6 câu điểm 2, 1 câu điểm 0). Tổng chi phí LLM cho mọi lần chạy khoảng $0,35.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Agent v1 → v2: v1 hay quên `[id]` trong text, A3 từ chối nhầm 4/18; v2 trả lời theo từng ý có citation, từ chối nhầm còn 0/18 nhưng lại trả lời cả 3 câu thiếu evidence. v2 → v3: thêm verifier, abstention lại 3/3 và từ chối nhầm vẫn 0/18, đổi lại chi phí tăng khoảng 50% và độ trễ trung vị từ 2,0 s lên 3,2 s.
  - Test: grounded trung bình 1,14 → 1,71, từ chối nhầm 2/7 → 0/7, abstention vẫn 1/2.
  - Giới hạn: test chỉ 7 + 2 câu (một câu = 14% hoặc 50%); dev và test cùng một sách; verifier dùng cùng model nên vẫn để lọt lỗi (q16); A3 trích thừa citation (6/13 hợp lệ); nhãn do một người chấm có AI hỗ trợ đọc trace; q22 và q10 có evidence thay thế mà metric nguyên văn không bắt nên phải chấm tay. Giá API theo giá niêm yết ($0,40/$1,60 cho 1 triệu token vào/ra).

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
