# 🛰️ Convex Hull Visualizer

An interactive **Convex Hull Visualizer** built using **Python and Streamlit**, showcasing the implementation and simulation of three major algorithms: **Graham Scan**, **QuickHull**, and **Jarvis March**.

This tool allows users to visualize how convex hulls are formed from a set of 2D points, either manually or through uploaded CSV files — even with **geographic coordinates** (latitude & longitude) for **real-world mapping**.

---

## 🚀 Features

- ✨ Add points interactively on a 2D Cartesian plane.
- 📁 Upload CSV files (with support for geographic data using `lat` and `lon` columns).
- 🔄 Choose between **Graham Scan**, **QuickHull**, and **Jarvis March** algorithms.
- 🔬 **Step-by-step simulation** of each algorithm's working.
- 🌍 **Geographic Hull Mode**: Visualize the convex boundary for real-world coordinates (e.g., state borders, delivery zones).
- 📊 Displays useful metrics like number of hull points, algorithm steps, and runtime.

---

## 🧠 Algorithms Implemented

### 📐 Graham Scan
- Sorts points by polar angle with respect to the lowest point.
- Uses a stack to construct the convex hull efficiently.
- Time Complexity: **O(n log n)**

### ⚡ QuickHull
- Divide-and-conquer method similar to QuickSort.
- Finds extreme points, recursively constructs upper and lower hulls.
- Average Time Complexity: **O(n log n)**

### 🧭 Jarvis March (Gift Wrapping)
- Starts from the leftmost point and "wraps" around the points like a gift.
- Best for small datasets.
- Time Complexity: **O(nh)** (where *h* is the number of hull points)

---

## 🌍 Real-World Applications

- 🗺️ **GIS and Geospatial Analysis**  
- 🤖 **Robotics & Pathfinding**
- 🧬 **Machine Learning (Clustering, Outlier Detection)**
- 🎮 **Game Development (Collision Detection)**
- 🧠 **Educational Tools for Algorithm Learning**

---

## 🛠️ Tech Stack

| Layer       | Technology        |
|-------------|-------------------|
| Frontend UI | Streamlit         |
| Backend     | Python            |
| Data Viz    | Matplotlib / Plotly |
| Data Handling | Pandas, NumPy     |
| Geo Support | CSV + Lat/Lon Columns (manual mapping or with `folium` support) |

---

## 📁 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/VIKASKT07/Convex-Hull-Visualizer.git
   pip install requirements.txt
   cd Convex-Hull-Visualizer
