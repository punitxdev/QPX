# Installation Guide

Thank you for choosing **KPX**. This guide will walk you through the installation process to get KPX Tabular up and running on your machine.

## Prerequisites

KPX requires **Python 3.8 or higher**. Before installing, ensure you have Python installed by running:

```bash
python --version
```

### Core Dependencies

KPX heavily relies on the following core libraries for its data manipulation and calculation capabilities:
- `pandas`
- `numpy`

*(Note: These will be installed automatically when you install KPX via pip.)*

## Installation Methods

### 1. Install via pip (Recommended)

If KPX is published to PyPI or you have a compiled wheel, you can install it directly using pip:

```bash
pip install kpx
```

### 2. Install from Source (Development)

If you want to try the latest development version, or if you plan to contribute to the KPX project, you can install it from source.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/punit-kashyap/kpx.git
   cd kpx
   ```

2. **Install the package in editable mode**:
   Installing in editable mode (`-e`) means that any changes you make to the source code will immediately take effect without needing to reinstall the package.
   ```bash
   pip install -e .
   ```

## Verifying the Installation

To verify that KPX was installed successfully, open a Python interactive shell or a Jupyter Notebook and try importing it:

```python
import pandas as pd
from kpx import Tabular

# Create a simple test dataframe
df = pd.DataFrame({"Test_Col": [1, 2, 3]})

# Initialize the Tabular class
tab = Tabular(df)

print("KPX Tabular loaded successfully!")
```

If the code runs without any `ModuleNotFoundError`, you are ready to go!

## Next Steps

Now that you have KPX installed, head over to the [Index](index.md) to learn about the core philosophy of KPX and explore its powerful automated preprocessing features.
