<p align="center">
  <img src="https://raw.githubusercontent.com/punitxdev/QPX/main/docs/assets/qpx_logo.png" alt="QPX Logo" width="350"/>
</p>

# QPX Tabular
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/Coverage-61%25-yellow.svg)]()
[![Documentation](https://img.shields.io/badge/Docs-Live-blue.svg)](https://punitxdev.github.io/QPX/)

QPX Tabular is a powerful, production-ready tabular data preprocessing and visualization library designed to accelerate data science workflows. It turns raw, messy pandas DataFrames into machine-learning ready datasets with a single line of code.

## Features

*   **Automated Preprocessing (`auto_preprocess`)**: Automatically handles missing values, drops constants, drops high-cardinality nominals, encodes categoricals intelligently, and downcasts memory.
*   **Fail-Loud Architecture**: Built for production. Instead of failing silently, QPX immediately alerts you (`KeyError`, `ValueError`) if you provide invalid data configurations.
*   **Comprehensive Data Health Diagnostics**: Get 360-degree views of your dataset's health via `dataset_health` and `statistical_snapshot`.
*   **Beautiful Visualizations**: One-line correlation heatmaps, distribution plots, and hierarchical feature clustering matrices.

## Installation

To install `qpx-tabular` via PyPI (once published) or from source, you can simply clone this repository and install it locally using `pip`:

```bash
git clone https://github.com/punitxdev/QPX.git
cd QPX
pip install -e .
```

### Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

## Quickstart

Clean an entire dataset with one function:

```python
import pandas as pd
from qpx.tabular import preprocessing

# Load your raw data
df = pd.read_csv("my_messy_data.csv")

# Clean, encode, impute, and downcast in one go!
clean_df, report = preprocessing.auto_preprocess(
    df,
    max_onehot=10, 
    return_report=True
)

print(report)
```

Generate a deep-dive correlation map:

```python
from qpx.tabular import visuals

visuals.corr_map(clean_df, target="my_target_column")
```

## Documentation
The complete API reference and user guide is hosted online at: 
**[https://punitxdev.github.io/QPX/](https://punitxdev.github.io/QPX/)**

If you want to build the documentation locally for development:

```bash
pip install -e .[dev]
mkdocs serve
```

To publish the documentation to GitHub Pages, simply run:
```bash
mkdocs gh-deploy
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Made with love by Punit
