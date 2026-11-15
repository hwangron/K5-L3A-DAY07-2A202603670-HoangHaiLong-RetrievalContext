# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: OpenStax Calculus Volume 1 (2018), mục 2.3, 2.4, 3.1, 3.2. Giấy phép CC BY-NC-SA 4.0. 67 trang PDF. 144 chunk lấy từ chiến lược C của Day 6 (`data/day6/chunks.json`).
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16 + pgvector 0.8.7 (Docker của Day 6, bảng `day6_chunks`, strategy C). Embedding `text-embedding-3-small`, 1536 chiều. Query được embed bằng đúng model này. Code kiểm tra model và số chiều trước khi tìm (`src/store.py`).
- Agent model, cách chạy và phân chia dev/test: `gpt-4.1-mini`, temperature 0, dùng tool calling của OpenAI. Tổng 30 câu. Dev có 18 câu có đáp án và 3 câu thiếu evidence. Test có 7 câu có đáp án và 2 câu thiếu evidence (`data/questions.json`).
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
  - Câu có đáp án lấy từ benchmark Day 6. Gold gồm nguồn, trang và trích dẫn nguyên văn. Trích dẫn phải nằm trọn trong một chunk (so sau khi gộp khoảng trắng, không phân biệt hoa thường), đã kiểm tra 0 lỗi.
  - Câu thiếu evidence hỏi về chủ đề gần nhưng nằm ngoài 4 mục (L'Hôpital, chain rule, epsilon-delta, product rule, năm xuất bản của Leibniz). Đã tìm từ khóa trên cả 67 trang (`data/unanswerable.json`).
  - Câu trả lời chấm 0/1/2 theo BENCHMARK.md, chấm tay bằng cách đọc trace.
  - Test của Day 6 chỉ chạy một lần ở Day 6, không dùng để chỉnh. Ở Day 7 tôi không xem test cho đến khi chốt cấu hình (commit `9599501`).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector, top-3, filter người dùng, không rerank, 800 ký tự, giữ chunk đến trước (A0) | Mặc định của starter | dev | 9/18 (50%) | 16/18 (89%) | 5/18 (28%); agent v1: 3/18 | 238 ms/truy vấn; agent $0,0013/câu |
| Phương pháp 2 | BM25 (k1 1,5, b 0,75), top-3, 800 ký tự | Từ khóa chính xác (conjugate, square root) giúp tìm đúng | dev | 12/18 (67%) | 17/18 (94%) | 6/18 (33%) | 0,3 ms; $0 |
| Phương pháp 3 | Hybrid RRF: dense top-20 + BM25 top-20, k = 60, top-3, 800 ký tự | Kết hợp hai cách để bù lỗi cho nhau | dev | 12/18 (67%) | 18/18 (100%) | 6/18 (33%) | dense + 4 ms; $0 thêm |
| Cải thiện 1 | Dense, top-3, 3000 ký tự, xếp rank-1 của mọi lượt tìm trước (A1) | Budget 800 không chứa nổi chunk ~620–1000 ký tự | dev | 9/18 (50%) | 16/18 (89%) | 16/18 (89%); agent v3: 14/18 | agent $0,0015/câu, trung vị 3,4 s |
| Cải thiện 2 | Hybrid RRF + rerank cross-encoder `ms-marco-MiniLM-L6-v2` trên 20 ứng viên, 3000 ký tự (A2) | Rerank đưa evidence lên hạng 1 | dev | 12/18 (67%) | 17/18 (94%) | 17/18 (94%); agent v3: 16/18 | +771 ms/truy vấn (CPU); agent $0,0013/câu |
| Cấu hình đã chọn | Hybrid RRF, top-3, 3000 ký tự, agent v3 (A3) | Chốt trên dev, xem bên dưới | test | 4/7 (57%) | 7/7 (100%) | 7/7 (100%); agent: 6/7 (86%) | agent $0,0015/câu, trung vị 3,5 s |
| Baseline | Như baseline ở dev, agent v1 (A0) | | test | 4/7 (57%) | 7/7 (100%) | 4/7 (57%); agent: 4/7 (57%) | agent $0,0012/câu, trung vị 4,6 s |

Cách đọc số liệu:
- Hit@1 và Hit@3 dùng câu hỏi gốc làm query. Context Evidence Rate có hai số:
  - Số đầu: context ghép từ top-3 của một lượt tìm.
  - Số sau "agent": context cuối mà agent gửi model, gồm cả các query do agent tự viết lại.
- Thời gian đo trên máy cá nhân. Dense gồm cả 1 lần gọi API embedding. Embedding query tốn dưới $0,00001 cho mỗi lần chạy. Trên dev, A3 đạt Hit@3 18/18 và Context agent 16/18.

Hai lỗi dev:
1. **Budget 800 ký tự.** A0 chỉ có 3/18 câu đủ evidence trong context. Chunk C dài trung vị 620 ký tự, cộng header thì gần như chỉ chứa được 1 chunk, nên agent tìm 3 lần mà vẫn thiếu. Tăng lên 3000 ký tự (A1) thì đạt 14/18.
2. **Dense trượt câu có từ khóa riêng, BM25 trượt câu dùng ký hiệu.** Dense trượt q04 (conjugate) và q06. BM25 trượt q02, vì sách viết "M ≠ 0" chứ không viết "not zero". Hybrid RRF tìm được cả ba (Hit@3 18/18). Rerank làm mất lại q02: cross-encoder chấm chunk ví dụ K/0 cao hơn chunk chứa định lý.

Ngoài ra có hai lỗi ở tầng agent, sửa bằng agent v2 và v3 (xem phần dưới).

Lý do chọn: theo tiêu chí đã chốt (`results/selection_criteria.md`), A3 có grounded trung bình cao nhất trên dev (1,94 so với 1,89 của A2), không cần reranker và nhanh hơn A2.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - **Tool và số lượt gọi.** Có 2 tool: `search(query)` và `answer(parts, abstain, missing)`. Lượt đầu bắt buộc gọi search. Tối đa 3 lần search, sau đó model chỉ còn tool answer.
  - **Dừng.** Agent dừng khi gọi answer. Nếu chạm giới hạn số bước thì từ chối.
  - **Context.** Chunk của mọi lượt tìm được bỏ trùng và xếp theo hạng. Context đóng gói trong 3000 ký tự. Model chỉ thấy context mới nhất, các context cũ thay bằng một dòng ngắn để không nhân budget lên.
  - **Citation.** Citation phải là ID có trong context. Một verifier (cùng model) kiểm tra từng ý có được chunk trích nêu ra không. Sai thì cho sửa 1 lần, vẫn sai thì từ chối. Khi từ chối, agent luôn trả câu cố định "Không đủ bằng chứng…".
  - **Lỗi.** Lỗi API hoặc DB được ghi `status: error`, không thay bằng kết quả giả. Không có lỗi nào trong các lần chạy.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A0 baseline, agent v1 / dev | 21/21 | 1 lần: 3, 3 lần: 18 | 9/3/6, 0,83 | 11/17 | 16/16 | 1/3 | 2/18 |
| A1, agent v3 / dev | 21/21 | 1: 15, 3: 6 | 2/0/16, 1,78 | 21/21 | 16/16 | 3/3 | 2/18 |
| A2, agent v3 / dev | 21/21 | 1: 17, 3: 4 | 1/0/17, 1,89 | 20/21 | 18/18 | 3/3 | 0/18 |
| A3, agent v3 / dev (chọn) | 21/21 | 1: 16, 2: 2, 3: 3 | 0/1/17, 1,94 | 23/23 | 18/18 | 3/3 | 0/18 |
| A0 baseline, agent v1 / test | 9/9 | 1: 4, 3: 5 | 3/0/4, 1,14 | 4/7 | 6/6 | 1/2 | 2/7 |
| A3, agent v3 / test (chọn) | 9/9 | 1: 6, 2: 1, 3: 2 | 1/0/6, 1,71 | 6/13 | 8/8 | 1/2 | 0/7 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q09 (A3, test) | Có đáp án | 1 search "ancient mathematician estimated area of circle using polygons", p183-00 ở hạng đầu | "Archimedes…" [p183-struct-00] | 2. Chunk nêu đúng Archimedes |
| q09 (A0, test) | Có đáp án | 3 search, p183-00 dài hơn 800 ký tự nên context rỗng | Từ chối | 0. Lỗi do budget, từ chối nhầm |
| u04 (A3, test) | Thiếu evidence | 3 search "product rule…", chỉ thấy chunk về đạo hàm khác. Verifier bác câu trả lời product rule | Từ chối | Đúng. Trong khi đó A0 v1 trả lời product rule từ kiến thức ngoài, trích chunk p240 không liên quan |
| q16 (A3, test) | Có đáp án | 2 search, tìm được đề Example 2.37 (p197-01) nhưng không có lời giải p198-00 | "IVT không chứng minh f có nghiệm" [p197-struct-01] | 0. Hiểu sai câu hỏi, chunk trích chỉ là đề bài. Chunking Day 6 tách đề và lời giải; verifier để lọt |
| u05 (A3, test) | Thiếu evidence | 3 search "Leibniz … publication year" | Gọi answer với câu "tài liệu không nêu năm" kèm 4 citation | Sai. Không bịa năm, nhưng không gọi abstain |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Retrieval: `results/retrieval/<split>/` (trace từng phương pháp và `summary.json`).
  - Agent: `results/agent/v<phiên bản>/<cấu hình>_<split>/`. Mỗi thư mục có `traces.json` (tool call, query, kết quả, context cuối, câu trả lời, token, thời gian), `labels.json` (nhãn tay và lý do), `reviews.json` và `summary.json` (tạo bằng `starter/agent_metrics.py`).
  - Điểm context 0/1/2 trên test: `results/agent/test_context_grades.json`. Baseline có 4 câu điểm 2 và 3 câu điểm 0; cấu hình chọn có 6 câu điểm 2 và 1 câu điểm 0.
  - Lệnh chạy ở `src/README.md`.
  - Tổng chi phí LLM cho mọi lần chạy khoảng $0,35. Embedding gần $0.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - **Agent v1 → v2.**
    - Lỗi v1: model thường quên ghi `[id]` vào text. Có câu đã có evidence mà vẫn bị từ chối: A3 v1 từ chối nhầm 4/18.
    - Sửa ở v2: model trả lời theo từng ý, mỗi ý kèm citation, nên format luôn đúng. Từ chối nhầm còn 0/18.
    - Lỗi mới ở v2: model gắn chunk bất kỳ vào lời giải thích lấy từ kiến thức ngoài. A3 v2 trả lời cả 3/3 câu thiếu evidence.
  - **Agent v2 → v3.** Thêm verifier. A3 quay lại từ chối đúng 3/3 mà vẫn 0/18 từ chối nhầm. Đổi lại, tốn thêm 1 lần gọi model, chi phí tăng khoảng 50% và độ trễ trung vị từ 2,0 s lên 3,2 s.
  - **Trên test.**
    - Grounded trung bình tăng từ 1,14 lên 1,71 và từ chối nhầm giảm từ 2/7 xuống 0/7. Abstention vẫn 1/2 ở cả hai cấu hình.
    - Citation hợp lệ của A3 thấp (6/13), vì model trích thừa: q12 trích 3 chunk, u05 trích 4 chunk. Verifier không bắt được lỗi này.
  - **Giới hạn.**
    - Test nhỏ: 1 câu bằng 14%.
    - Dev và test cùng một cuốn sách.
    - Metric so trích dẫn nguyên văn bỏ sót evidence thay thế, ví dụ q22 p244 và q10 nằm ở 2 chunk. Vì vậy cần chấm tay.
    - Verifier dùng cùng model nên vẫn để lọt lỗi (q02 ở A2, q16).
    - Nhãn do một người chấm, có AI hỗ trợ đọc trace.
    - Giá `gpt-4.1-mini` lấy theo giá niêm yết: $0,40 / $1,60 cho mỗi 1 triệu token vào / ra.

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: Không thực hiện.
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý: Không thực hiện.
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev: Không thực hiện.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống: Không thực hiện.
- Giới hạn, cách tái lập và đường dẫn benchmark: Không thực hiện.
