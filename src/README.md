# Code Day 7

| File | Việc |
|---|---|
| `common.py` | Đường dẫn, config, đọc dữ liệu, so evidence (`is_hit`), đóng gói context |
| `store.py` | Đọc vector store Day 6 (pgvector, chiến lược C) và tạo query embedding cùng model |
| `export_day6.py` | Xuất 144 chunk C từ pgvector, chuyển benchmark Day 6 sang schema Day 7 |
| `retrievers.py` | 4 phương pháp: `dense`, `bm25`, `hybrid_rrf`, `hybrid_rerank` |
| `eval_retrieval.py` | Đo Hit@1, Evidence Hit@3, Context Evidence Rate với query gốc |
| `agent.py` | Agent dùng tool calling (`search`, `answer`) với model OpenAI |
| `run_agent.py` | Chạy agent theo cấu hình trên một split, lưu trace |
| `make_reviews.py` | Ghép nhãn chấm tay thành `reviews.json` cho `starter/agent_metrics.py` |

## Cách chạy

Chạy từ thư mục gốc của repo, Python 3.11 trở lên.

```bash
pip install -r requirements.txt
cp .env.example .env            # điền DATABASE_URL và OPENAI_API_KEY
# Bật pgvector của Day 6 (trong thư mục starter của repo Day 6):
docker compose -p day6_student_starter up -d

python -m src.export_day6 --day6 <thư mục repo Day 6>
python -m unittest discover -s tests -v
python -m src.eval_retrieval --split dev
# --agent-version: 1 = bản đầu, 2 = trả lời theo từng ý, 3 = thêm verifier (mặc định)
python -m src.run_agent --config A0_dense_starter --split dev --agent-version 1
python -m src.run_agent --config A1_dense_budget --split dev
python -m src.run_agent --config A2_hybrid_rerank --split dev
python -m src.run_agent --config A3_hybrid_rrf --split dev
# sau khi viết labels.json bằng tay trong thư mục kết quả:
python -m src.make_reviews --config <config> --split dev --agent-version <v>
python starter/agent_metrics.py --cases data/cases.json --traces results/agent/v<v>/<config>_dev/traces.json \
  --reviews results/agent/v<v>/<config>_dev/reviews.json --split dev --output results/agent/v<v>/<config>_dev/summary.json

# test, chỉ chạy một lần sau khi chốt cấu hình (results/selection_decision.md):
python -m src.eval_retrieval --split test
python -m src.run_agent --config A0_dense_starter --split test --agent-version 1   # baseline
python -m src.run_agent --config A3_hybrid_rrf --split test --agent-version 3      # cấu hình chọn
```

Lần chạy đầu tiên của `hybrid_rerank` cần mạng để tải model rerank `cross-encoder/ms-marco-MiniLM-L6-v2` (không phải gói pip, không nằm trong `requirements.txt`) từ HuggingFace vào cache; các lần sau dùng lại.

Model dùng `temperature 0`, nhưng kết quả chạy lại vẫn có thể lệch đôi chút. Kết quả đã chấm nằm trong `results/`.

Nếu Postgres chưa chạy hoặc API lỗi, chương trình báo lỗi. Trong `run_agent`, câu bị lỗi được ghi `status: "error"`, không thay bằng kết quả giả. Embedding của query được cache trong `.cache/` để chạy lại không tốn thêm phí.
