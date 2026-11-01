"""Root entrypoint to run quantum circuit time estimation experiments using in-code configuration."""

from qcirc_time_estimator.main import main
from qcirc_time_estimator.config import (
    ACTIVE_CONFIG,
    MULTI_BACKEND_SAMPLE_CONFIG,
    FAST_MULTI_BACKEND_CONFIG,
    FULL_68_BACKENDS_CONFIG,
    FAST_TEST_CONFIG,
    FULL_MQT_BENCHMARK_CONFIG,
    SINGLE_CIRCUIT_CONFIG,
)

if __name__ == "__main__":
    # You can edit ACTIVE_CONFIG in src/qcirc_time_estimator/config.py
    # or pass a custom ExperimentConfig directly: main(FAST_MULTI_BACKEND_CONFIG)
    main(ACTIVE_CONFIG)
