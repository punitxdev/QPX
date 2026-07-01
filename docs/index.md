# Welcome to QPX

<p align="center">
  <img src="assets/qpx_logo.png" alt="QPX Logo" width="350"/>
</p>

**QPX (Kashyap Preprocessing Xpress)** is an advanced, automated machine learning and data processing toolkit designed to simplify and accelerate the modern data science workflow. 

Instead of writing hundreds of lines of boilerplate code for every new project, QPX provides you with robust, production-ready pipelines that clean, diagnose, and optimize your data in just a few lines of code. **Built on top of industry-standard libraries like `pandas`, `numpy`, `scikit-learn`, and `matplotlib`/`seaborn`, QPX drastically reduces developer effort and accelerates the entire data preprocessing lifecycle.**

---

## Why QPX?

The modern data science pipeline is often bogged down by repetitive, manual tasks. QPX solves this through **intelligent automation with complete transparency**.

- **Smart Automation**: Automatically detect disguised numeric columns, drop useless zero-variance features, and fix high-cardinality issues without lifting a finger.
- **Safe & Transparent**: Every automated pipeline returns a comprehensive JSON-like report detailing exactly what transformations were applied to your dataset.
- **Memory Optimized**: Automatically downcasts data types to safely slash your dataset's memory footprint by up to 50% or more, crucial for large-scale processing.
- **ML Ready**: One-line diagnostic checks to instantly tell you if your data is structurally safe to feed into a machine learning algorithm.

---

## What's in the Box?

QPX is built as a modular framework. Its flagship module is **QPX Tabular** (`qpx.tabular`), a comprehensive suite dedicated to tabular data (like CSVs, SQL tables, and Excel sheets). 

It unifies three critical areas of data preparation:

### 1. [Analytical Profiling](user_guide/tabular/summary.md)
Instantly assess dataset health, structural integrity, and machine learning readiness. Stop guessing where your missing values and messy data types are hiding.

### 2. [Statistical Diagnostics](user_guide/tabular/statistics.md)
Dive deep into mathematical properties. Automatically detect statistical outliers (using IQR), measure dataset entropy, and generate Variance Inflation Factor (VIF) reports to prevent multicollinearity.

### 3. [Automated Preprocessing](user_guide/tabular/preprocessing.md)
End-to-end data cleaning. A single, highly configurable pipeline that handles everything from snake_case column sanitization to smart NaN imputation and automated categorical encoding (Label, One-Hot, Hash).

---

## Where to go next?

Ready to stop cleaning data manually? Here is how to get started:

- **[Installation Guide](installation.md)**: How to install QPX into your environment.
- **[5-Minute Quickstart](quickstart.md)**: See QPX in action and learn how to run the flagship pipeline on your own data immediately.
- **[Tabular Overview](user_guide/tabular/overview.md)**: A deeper dive into how the `Tabular` wrapper class works under the hood.

---

## About the Author

**Punit Kashyap** is the creator and lead developer of QPX. Currently a student at **IIT Dharwad**, Punit has a strong focus on artificial intelligence, data engineering, and intelligent automation. 

Dedicated to creating institutional-quality software, QPX is a reflection of Punit's commitment to technical rigor, efficiency, and developer experience. The goal of QPX is to push the boundaries of algorithmic processing and provide professionals with highly optimized, user-friendly tools.

---
*Built with passion to accelerate preprocessing, drastically reduce model development time, and make data science easier for everyone.*
