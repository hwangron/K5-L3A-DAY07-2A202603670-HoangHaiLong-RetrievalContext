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
python -m src.run_agent --config A0_dense_starter --split dev
python -m src.run_agent --config A1_dense_budget --split dev
python -m src.run_agent --config A2_hybrid_rerank --split dev
python -m src.run_agent --config A3_hybrid_rrf --split dev
python -m src.make_reviews --config <config> --split dev
python starter/agent_metrics.py --cases data/cases.json --traces results/agent/<config>_dev/traces.json \
  --reviews results/agent/<config>_dev/reviews.json --split dev --output results/agent/<config>_dev/summary.json
```

Bước test làm giống bước dev, đổi `--split test`. Chỉ chạy sau khi đã chốt cấu hình.

Nếu Postgres chưa chạy hoặc API lỗi, chương trình báo lỗi. Trong `run_agent`, câu bị lỗi được ghi `status: "error"`, không thay bằng kết quả giả. Embedding của query được cache trong `.cache/` để chạy lại không tốn thêm phí.
