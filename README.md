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

## 2. Kiến Trúc Khung Repo (Mô-đun Hóa & Chuẩn Hóa)

```text
estimated_time/
├── main.py                            # Entry point thực thi thực nghiệm trực tiếp
├── data/
│   ├── circuits/                      # Cache các mạch QASM
│   └── results/                       # Bảng CSV dataset, metadata và metrics kết quả
│       ├── ground_truth_dataset.csv   # Dataset chuẩn Ground Truth 51 cột cho ML
│       └── backends_metadata.csv      # Thông số kỹ thuật các backend IBM
├── paper/                             # Tài liệu báo cáo / bản thảo bài báo
│   ├── paper.tex                      # Bản thảo LaTeX bài báo khoa học
│   └── paper.pdf                      # Bản PDF bài báo khoa học đã biên dịch
├── src/
│   └── qcirc_time_estimator/
│       ├── backends/                  # Quản lý 68 fake backends và BackendInfo
│       │   ├── info.py                # Dataclass BackendInfo & ma trận khoảng cách
│       │   ├── provider.py            # IBM FakeProvider integration & extractor
│       │   └── __init__.py            # Export module backends
│       ├── circuits/                  # Nạp mạch từ MQT Bench theo dải qubit
│       │   ├── loader.py              # Loader & benchmark suite generator
│       │   └── __init__.py            # Export module circuits
│       ├── ground_truth/              # Bộ tính duration chính xác bằng Qiskit
│       │   └── scheduler.py           # Transpile + pulse scheduling (ASAP/ALAP)
│       │   └── __init__.py            # Export module ground_truth
│       ├── features/                  # Trích xuất đặc trưng mạch & phần cứng
│       │   ├── extractor.py           # phi_circuit, phi_machine & tương tác
│       │   └── __init__.py            # Export module features
│       ├── dataset/                   # Quản lý lưu trữ dataset và metadata
│       │   ├── manager.py             # CSVDataManager cho dataset 51 cột
│       │   └── __init__.py            # Export module dataset
│       ├── estimators/                # Các mô hình ước tính thời gian trước biên dịch
│       │   ├── base.py                # Abstract class BaseDurationEstimator
│       │   ├── analytical.py          # Critical path DAG traversal
│       │   ├── routing_aware.py       # Critical path + SWAP routing penalty
│       │   ├── clops_depth_estimator.py # Ước tính NewDepth * CLOPS (Physics & Calibrated)
│       │   ├── ml_estimator.py        # Random Forest Regressor trên vector đặc trưng
│       │   └── __init__.py            # Export module estimators
│       ├── evaluation/                # Điều phối thực nghiệm, đo lường & biểu đồ
│       │   ├── runner.py              # BenchmarkRunner & MultiBackendRunner
│       │   ├── metrics.py             # Tính toán MAE, MAPE, RMSE, Pearson r, Speedup
│       │   ├── visualizer.py          # Xuất đồ thị Scatter, Error Bar và Speedup
│       │   └── __init__.py            # Export module evaluation
│       ├── config.py                  # Cấu hình thực nghiệm & các preset
│       ├── main.py                    # Dispatcher chính của package
│       ├── __main__.py                # Cho phép python -m qcirc_time_estimator
│       └── __init__.py                # Export public API tinh gọn, chuẩn hóa
├── tests/                             # Bộ unit test toàn diện (pytest - 25 tests)
│   ├── test_backend.py
│   ├── test_mqt_loader.py
│   ├── test_ground_truth.py
│   ├── test_estimators.py
│   ├── test_clops_depth_estimator.py
│   ├── test_benchmark.py
│   ├── test_features.py
│   ├── test_csv_manager.py
│   └── test_multi_backend.py
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
| **CLOPS-Aware NewDepth** | Ước tính qua công thức $\text{NewDepth} \times \tau_{\text{layer}}$ (với $\tau_{\text{layer}} = 1/\text{CLOPS} \approx t_{CX}$), trong đó $\text{NewDepth} = D_0 + \alpha \Delta_{1Q} + \beta \Delta_{\text{topo}} + \gamma \Delta_{\text{multi}} + \mu \Delta_{\text{meas}}$. Hỗ trợ cả 2 chế độ: Physics-Informed (Zero-shot) và Calibrated (NNLS). | Cực nhanh ($> 150\text{x}$ speedup), mô hình hóa tường minh các yếu tố dãn nở độ sâu vật lý, tương quan cao ($r > 0.82$). |
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

### 4.3. Cấu Hình Thực Nghiệm Trực Tiếp Trong Code (`config.py`)
Toàn bộ cấu hình thực nghiệm được quản lý bằng Python code trong `src/qcirc_time_estimator/config.py`. Bạn không cần phải nhớ hay gõ bất kỳ cờ dòng lệnh CLI phức tạp nào.

Bạn có thể chỉnh sửa `ACTIVE_CONFIG` hoặc tùy biến các tham số trong `ExperimentConfig`:
```python
from qcirc_time_estimator.config import ExperimentConfig

my_config = ExperimentConfig(
    mode="benchmark",  # hoặc "single_circuit"
    benchmarks=["ghz", "qft", "grover"],
    qubit_range=(2, 4),
    optimization_level=1,
    output_dir="data/results/my_run",
)
```

Các preset có sẵn trong code:
- **`FAST_TEST_CONFIG`**: Chạy nhanh trên 3 thuật toán (`ghz, qft, grover`) từ 2 đến 3 qubit.
- **`FULL_MQT_BENCHMARK_CONFIG`**: Chạy toàn diện 8 thuật toán trên dải 2..5 qubit.
- **`SINGLE_CIRCUIT_CONFIG`**: Đánh giá chi tiết cho 1 mạch đơn lẻ (ví dụ QFT 3 qubit).

### 4.4. Chạy Thực Nghiệm (Không Cần Parser)
Chạy thực nghiệm trực tiếp chỉ với một lệnh:
```bash
python main.py
# hoặc
python -m qcirc_time_estimator
```

Kết quả xuất ra (khi chạy ở chế độ benchmark):
- File dữ liệu chi tiết: `data/results/<output_dir>/benchmark_results.csv`
- Bảng tổng kết metrics: `data/results/<output_dir>/summary_metrics.json`
- Biểu đồ tương quan: `data/results/<output_dir>/correlation_plot.png`
- Biểu đồ sai số theo thuật toán: `data/results/<output_dir>/error_by_algorithm.png`
- Biểu đồ hệ số tăng tốc: `data/results/<output_dir>/speedup_plot.png`

Khi chạy ở chế độ `single_circuit`, kết quả so sánh chi tiết giữa Ground Truth và các Estimators (kèm theo phân rã $D_0, \Delta_{\text{1Q}}, \Delta_{\text{topo}}, \text{NewDepth}$) sẽ được in trực tiếp ra màn hình console.

---

## 5. Các Chỉ Số Đánh Giá (Metrics)

Khung thực nghiệm tự động tính toán các chỉ số thống kê chuẩn:
- **MAPE (%)**: Sai số phần trăm tuyệt đối trung bình $\frac{100\%}{N}\sum \frac{|T_{\text{est}} - T_{\text{true}}|}{T_{\text{true}}}$.
- **MAE (μs)**: Sai số tuyệt đối trung bình $\frac{1}{N}\sum |T_{\text{est}} - T_{\text{true}}|$.
- **RMSE (μs)**: Căn bậc hai sai số bình phương trung bình.
- **Pearson $r$**: Hệ số tương quan tuyến tính giữa giá trị ước tính và giá trị thực tế.
- **Spearman $\rho$**: Hệ số tương quan thứ bậc (đánh giá khả năng xếp hạng độ dài mạch).
- **Speedup Factor**: Tỉ số giữa thời gian transpile của Qiskit và thời gian suy luận của Estimator ($t_{\text{transpile}} / t_{\text{est}}$).

