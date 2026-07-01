# KPX Tabular
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)]()

KPX Tabular is a powerful, production-ready tabular data preprocessing and visualization library designed to accelerate data science workflows. It turns raw, messy pandas DataFrames into machine-learning ready datasets with a single line of code.

## 🚀 Features

*   **Automated Preprocessing (`auto_preprocess`)**: Automatically handles missing values, drops constants, drops high-cardinality nominals, encodes categoricals intelligently, and downcasts memory.
*   **Fail-Loud Architecture**: Built for production. Instead of failing silently, KPX immediately alerts you (`KeyError`, `ValueError`) if you provide invalid data configurations.
*   **Comprehensive Data Health Diagnostics**: Get 360-degree views of your dataset's health via `dataset_health` and `statistical_snapshot`.
*   **Beautiful Visualizations**: One-line correlation heatmaps, distribution plots, and hierarchical feature clustering matrices.

## 📦 Installation

To install `kpx`, you can simply clone this repository and install it locally using `pip`:

```bash
git clone https://github.com/punitxdev/KPX.git
cd KPX
pip install -e .
```

### Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

## 💡 Quickstart

Clean an entire dataset with one function:

```python
import pandas as pd
from kpx.tabular import preprocessing

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
from kpx.tabular import visuals

visuals.corr_map(clean_df, target="my_target_column")
```

## 📚 Documentation
Full API references and user guides can be generated locally using MkDocs:

```bash
pip install -e .[dev]
mkdocs serve
```

To publish the documentation to GitHub Pages, simply run:
```bash
mkdocs gh-deploy
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Made with ❤️ by Punit
