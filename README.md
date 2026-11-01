# Quantum Circuit Execution Time Estimator on IBM FakeBelem

Một khung thực nghiệm (experimental benchmarking framework) độc lập, khoa học và mở rộng nhằm ước tính nhanh thời gian thực thi (latency / duration) của mạch lượng tử trên kiến trúc máy lượng tử IBM (cụ thể là **FakeBelem**), đối chiếu trực tiếp với giá trị chính xác (Ground Truth) thu được từ Qiskit Transpiler & Scheduler.

---

## 1. Giới Thiệu & Đặt Vấn Đề

Khi lập trình và tối ưu hóa giải thuật lượng tử, việc tính toán thời gian thực thi chính xác của một mạch lượng tử trên phần cứng vật lý thông thường đòi hỏi quy trình biên dịch hoàn chỉnh qua Qiskit Transpiler:
1. **Layout & Routing**: Ánh xạ qubit logic vào các qubit vật lý trên đồ thị ghép nối (Coupling Map) và chèn các cổng hoán đổi (SWAP) để tương thích tô-pô.
2. **Basis Translation**: Dịch các cổng logic về tập cổng cơ sở vật lý ($\{cx, id, rz, sx, x\}$ trên IBM Belem).
3. **Pulse Scheduling**: Lập lịch thời gian phát xung (ASAP hoặc ALAP) dựa trên chu kỳ đồng hồ $dt$ và độ trễ xung thực nghiệm của từng cổng.

Quy trình trên tiêu tốn đáng kể thời gian tính toán của CPU (từ hàng trăm mili-giây đến hàng giây cho mỗi mạch). **Mục tiêu của repository này** là cung cấp các mô hình ước tính siêu nhanh (thực thi dưới 1 mili-giây, tăng tốc hàng ngàn lần) ngay từ các mạch ở mức **`BenchmarkLevel.INDEP`** (mạch đã phân rã về cổng cơ bản nhưng độc lập phần cứng), đồng thời đo lường độ chính xác và tương quan của giá trị ước tính so với giá trị thực tế sau khi transpile.

---

## 2. Kiến Trúc Khung Repo

```text
estimated_time/
├── configs/
│   └── default_experiment.yaml        # Cấu hình danh sách benchmark, opt level, backend
├── data/
│   ├── circuits/                      # Cache các mạch QASM
│   └── results/                       # Bảng CSV, JSON metrics và biểu đồ PNG kết quả
├── src/
│   └── qcirc_time_estimator/
│       ├── backends/                  # Quản lý FakeBelemV2 và ma trận khoảng cách tô-pô
│       │   ├── backend_info.py        # Dataclass BackendInfo
│       │   └── fake_belem.py          # Trích xuất gate durations, dt, BFS shortest paths
│       ├── circuits/                  # Nạp mạch từ MQT Bench (Munich Quantum Toolkit)
│       │   └── mqt_loader.py          # Nạp mạch INDEP / ALG theo dải qubit 2..5
│       ├── ground_truth/              # Bộ tính duration chính xác bằng Qiskit
│       │   └── qiskit_scheduler.py    # Transpile + Scheduling (ASAP/ALAP)
│       ├── estimators/                # Các mô hình ước tính thời gian
│       │   ├── base.py                # Abstract class BaseDurationEstimator
│       │   ├── analytical.py          # Critical path DAG traversal
│       │   ├── routing_aware.py       # Critical path + SWAP routing penalty
│       │   └── ml_estimator.py        # Random Forest Regressor trên vector đặc trưng mạch
│       ├── evaluation/                # Bộ đo lường, benchmark runner và vẽ biểu đồ
│       │   ├── metrics.py             # MAE, MAPE, RMSE, Pearson r, Spearman rho, Speedup
│       │   ├── benchmark_runner.py    # Điều phối thực nghiệm tự động
│       │   └── visualizer.py          # Xuất đồ thị Scatter, Error Bar và Speedup
│       └── cli.py                     # Giao diện dòng lệnh CLI
├── tests/                             # Bộ unit test toàn diện (pytest)
│   ├── test_backend.py
│   ├── test_mqt_loader.py
│   ├── test_ground_truth.py
│   ├── test_estimators.py
│   └── test_benchmark.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Các Phương Pháp Ước Tính

| Phương Pháp | Mô Tả Kỹ Thuật | Đặc Điểm |
| :--- | :--- | :--- |
| **Ground Truth (Qiskit)** | `transpile(qc, backend=FakeBelemV2(), scheduling_method='asap')` $\to D_{\text{clock}} \times dt$ | Độ chính xác tuyệt đối trên mô hình giả lập, tốn thời gian CPU. |
| **Analytical Critical Path** | Duyệt DAG theo các đường qubit wire, tích lũy độ trễ của cổng 1Q và đồng bộ hóa ở các cổng 2Q. | Siêu nhanh ($< 0.1\text{ ms}$), phản ánh đúng đường tới hạn lý tưởng. |
| **Routing-Aware** | Tích hợp ma trận khoảng cách đường đi ngắn nhất $D_{ij}$ trên coupling map chữ T của Belem: cộng dồn $(D_{ij} - 1) \times 3 \times t_{CX}$ cho các cổng 2Q không kề nhau. | Giảm thiểu sai số do các cổng SWAP gây ra khi ánh xạ lên phần cứng thực. |
| **ML Surrogate Regressor** | Trích xuất 9 đặc trưng (độ sâu, số lượng 1Q/2Q/Meas, mật độ, khoảng cách tương tác, neo analytical) $\to$ huấn luyện RandomForestRegressor. | Học trực tiếp quan hệ phi tuyến giữa cấu trúc mạch và thời gian chạy. |

---

## 4. Hướng Dẫn Cài Đặt & Sử Dụng

### 4.1. Kích Hoạt Môi Trường Conda
Khung thực nghiệm được cấu hình chạy trên môi trường conda **`timeestimate`** (Python 3.10):
```bash
conda activate timeestimate
```

Nếu chưa cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
# hoặc
pip install -e .
```

### 4.2. Chạy Kiểm Thử Tự Động (Unit Tests)
Chạy toàn bộ test suite để xác nhận tính toàn vẹn của hệ thống:
```bash
pytest tests/ -v
```

### 4.3. Chạy Thực Nghiệm Benchmark MQT Bench (CLI)
Chạy thử nghiệm trên các thuật toán MQT Bench (`ghz, qft, dj, bv, grover, graphstate, wstate, ae`) với số qubit từ 2 đến 5 trên `FakeBelem`:
```bash
python -m qcirc_time_estimator.cli run-mqt-benchmark \
  --benchmarks ghz,qft,dj,bv,grover,graphstate,wstate,ae \
  --min-qubits 2 \
  --max-qubits 5 \
  --opt-level 1 \
  --level indep \
  --output-dir data/results/demo_run
```

Kết quả xuất ra:
- File dữ liệu chi tiết: `data/results/demo_run/benchmark_results.csv`
- Bảng tổng kết metrics: `data/results/demo_run/summary_metrics.json`
- Biểu đồ tương quan: `data/results/demo_run/correlation_plot.png`
- Biểu đồ sai số theo thuật toán: `data/results/demo_run/error_by_algorithm.png`
- Biểu đồ hệ số tăng tốc: `data/results/demo_run/speedup_plot.png`

### 4.4. Ước Tính Cho Một Mạch Lượng Tử Đơn Lẻ
Ước tính thời gian thực thi của một thuật toán cụ thể từ MQT Bench:
```bash
python -m qcirc_time_estimator.cli estimate --benchmark-name qft --num-qubits 4 --level indep --opt-level 1
```

Hoặc ước tính từ một file OpenQASM tùy ý:
```bash
python -m qcirc_time_estimator.cli estimate --qasm-file path/to/my_circuit.qasm --opt-level 1
```

---

## 5. Các Chỉ Số Đánh Giá (Metrics)

Khung thực nghiệm tự động tính toán các chỉ số thống kê chuẩn:
- **MAPE (%)**: Sai số phần trăm tuyệt đối trung bình $\frac{100\%}{N}\sum \frac{|T_{\text{est}} - T_{\text{true}}|}{T_{\text{true}}}$.
- **MAE (μs)**: Sai số tuyệt đối trung bình $\frac{1}{N}\sum |T_{\text{est}} - T_{\text{true}}|$.
- **RMSE (μs)**: Căn bậc hai sai số bình phương trung bình.
- **Pearson $r$**: Hệ số tương quan tuyến tính giữa giá trị ước tính và giá trị thực tế.
- **Spearman $\rho$**: Hệ số tương quan thứ bậc (đánh giá khả năng xếp hạng độ dài mạch).
- **Speedup Factor**: Tỉ số giữa thời gian transpile của Qiskit và thời gian suy luận của Estimator ($t_{\text{transpile}} / t_{\text{est}}$).

