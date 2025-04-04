import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.spatial import ConvexHull
import folium
from streamlit_folium import folium_static
from math import atan2

# --------------------- FUNCTION DEFINITIONS ---------------------
def orientation(p, q, r):
    val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
    if val == 0: return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def plot_current_state(ax, points, hull, algorithm):
    ax.plot(points[:,0], points[:,1], 'bo')
    if len(hull) > 1:
        ax.plot([p[0] for p in hull], [p[1] for p in hull], 'r-', lw=2)
    if hull:
        ax.plot(hull[-1][0], hull[-1][1], 'go', markersize=10)
    ax.set_title(f"{algorithm} - Step-by-Step")

def graham_scan(points):
    pivot = min(points, key=lambda p: (p[1], p[0]))
    sorted_points = sorted(points, key=lambda p: (atan2(p[1]-pivot[1], p[0]-pivot[0]), p[0], p[1]))
    hull = [pivot, sorted_points[0]]
    
    placeholder = st.empty()
    fig, ax = plt.subplots()
    
    for point in sorted_points[1:]:
        while len(hull) > 1 and orientation(hull[-2], hull[-1], point) != 2:
            hull.pop()
            ax.clear()
            plot_current_state(ax, points, hull, "Graham's Scan")
            placeholder.pyplot(fig)
            time.sleep(0.5)
            
        hull.append(point)
        ax.clear()
        plot_current_state(ax, points, hull, "Graham's Scan")
        placeholder.pyplot(fig)
        time.sleep(0.5)
    
    return np.array(hull)

def distance(p1, p2, p3):
    return abs((p2[1]-p1[1])*p3[0] - (p2[0]-p1[0])*p3[1] + p2[0]*p1[1] - p2[1]*p1[0]) / np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def quickhull_helper(points, p1, p2):
    if len(points) == 0:
        return []
    left_points = [p for p in points if orientation(p1, p2, p) == 2]
    if not left_points:
        return []
    farthest = max(left_points, key=lambda p: distance(p1, p2, p))
    hull = []
    hull.extend(quickhull_helper(left_points, p1, farthest))
    hull.append(farthest)
    hull.extend(quickhull_helper(left_points, farthest, p2))
    return hull

def quickhull(points):
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    left = points[np.argmin(points[:,0])]
    right = points[np.argmax(points[:,0])]
    hull = []
    hull.extend(quickhull_helper(points, left, right))
    hull.extend(quickhull_helper(points, right, left))
    center = np.mean(hull, axis=0)
    hull = sorted(hull, key=lambda p: -atan2(p[1]-center[1], p[0]-center[0]))
    return np.array(hull)

def jarvis_march(points):
    n = len(points)
    if n < 3:
        return points
    l = np.argmin(points[:,0])
    hull = []
    p = l
    q = 0
    
    placeholder = st.empty()
    fig, ax = plt.subplots()
    
    while True:
        hull.append(points[p])
        ax.clear()
        plot_current_state(ax, points, [points[i] for i in hull], "Jarvis March")
        placeholder.pyplot(fig)
        time.sleep(0.5)
        
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        p = q
        if p == l:
            break
    
    return np.array(hull)

def plot_hull(points, hull, algorithm):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:,0], points[:,1], 'bo', label='Points')
    ax.plot(hull[:,0], hull[:,1], 'r-', lw=2, label='Convex Hull')
    ax.plot([hull[-1,0], hull[0,0]], [hull[-1,1], hull[0,1]], 'r-', lw=2)
    ax.set_title(f"{algorithm} - Convex Hull")
    ax.legend()
    st.pyplot(fig)

def geographic_convex_hull(points):
    points_cartesian = np.column_stack([points[:,0], points[:,1]])
    hull_indices = ConvexHull(points_cartesian).vertices
    return points[hull_indices]

# --------------------- STREAMLIT UI CODE ---------------------
st.set_page_config(page_title="Convex Hull Visualizer", layout="wide")
st.title("Convex Hull Algorithm Visualizer")
st.write("A DAA project demonstrating convex hull algorithms with interactive visualizations")

if 'points' not in st.session_state:
    st.session_state.points = []
if 'geo_points' not in st.session_state:
    st.session_state.geo_points = []

tab1, tab2, tab3 = st.tabs(["2D Cartesian", "Geographical", "Algorithm Comparison"])

with tab1:
    st.header("2D Cartesian Convex Hull")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_method = st.radio("Input method:", 
                              ["Manual Entry", "Random Points", "Upload CSV"])
        
        if input_method == "Manual Entry":
            x = st.number_input("X coordinate", value=0.0)
            y = st.number_input("Y coordinate", value=0.0)
            if st.button("Add Point"):
                st.session_state.points.append((x, y))
                
        elif input_method == "Random Points":
            num_points = st.slider("Number of points", 5, 100, 20)
            if st.button("Generate Points"):
                st.session_state.points = np.random.rand(num_points, 2).tolist()
                
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.points = df.values.tolist()
                
        if st.button("Clear Points"):
            st.session_state.points = []
            
        algorithm = st.selectbox("Select algorithm:", 
                               ["Graham's Scan", "Quickhull", "Jarvis March"])
        
        if st.button("Compute Convex Hull"):
            points = np.array(st.session_state.points)
            if len(points) < 3:
                st.warning("Need at least 3 points to compute convex hull")
            else:
                with st.spinner("Computing convex hull..."):
                    start_time = time.time()
                    if algorithm == "Graham's Scan":
                        hull = graham_scan(points)
                    elif algorithm == "Quickhull":
                        hull = quickhull(points)
                    else:
                        hull = jarvis_march(points)
                    end_time = time.time()
                    
                st.success(f"Computed in {end_time - start_time:.4f} seconds")
                plot_hull(points, hull, algorithm)
    
    with col2:
        if st.session_state.points:
            fig, ax = plt.subplots()
            points = np.array(st.session_state.points)
            ax.plot(points[:,0], points[:,1], 'bo')
            ax.set_title("Current Points")
            st.pyplot(fig)

with tab2:
    st.header("Geographical Convex Hull")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Geographical Points")
        lon = st.number_input("Longitude", -180.0, 180.0, 77.2090)
        lat = st.number_input("Latitude", -90.0, 90.0, 28.6139)
        
        if st.button("Add Location"):
            st.session_state.geo_points.append([lon, lat])
            st.success(f"Added point: {lon}, {lat}")
            
        if st.button("Clear Locations"):
            st.session_state.geo_points = []
            
        if st.button("Compute Geographical Hull"):
            if len(st.session_state.geo_points) < 3:
                st.warning("Need at least 3 points to compute convex hull")
            else:
                points = np.array(st.session_state.geo_points)
                hull = geographic_convex_hull(points)
                
                m = folium.Map(location=[np.mean(points[:,1]), np.mean(points[:,0])], zoom_start=4)
                for point in points:
                    folium.Marker([point[1], point[0]]).add_to(m)
                hull_points = [[p[1], p[0]] for p in hull]
                folium.PolyLine(hull_points + [hull_points[0]], color="red", weight=2.5, opacity=1).add_to(m)
                folium_static(m, width=800, height=600)

with tab3:
    st.header("Algorithm Comparison")
    num_points = st.slider("Number of points for comparison", 10, 10000, 100)
    
    if st.button("Run Comparison"):
        points = np.random.rand(num_points, 2)
        results = []
        
        start = time.time()
        graham_scan(points.copy())
        results.append(("Graham's Scan", time.time() - start, "O(n log n)"))
        
        start = time.time()
        quickhull(points.copy())
        results.append(("Quickhull", time.time() - start, "O(n log n) average, O(n²) worst"))
        
        start = time.time()
        jarvis_march(points.copy())
        results.append(("Jarvis March", time.time() - start, "O(nh)"))
        
        start = time.time()
        ConvexHull(points.copy())
        results.append(("SciPy (Qhull)", time.time() - start, "O(n log n)"))
        
        df = pd.DataFrame(results, columns=["Algorithm", "Time (s)", "Time Complexity"])
        st.table(df)
        
        fig, ax = plt.subplots()
        ax.bar(df["Algorithm"], df["Time (s)"])
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("Algorithm Performance Comparison")
        st.pyplot(fig)
        
        st.markdown("""
        **Time Complexity Analysis:**
        - **Graham's Scan**: O(n log n) due to sorting step
        - **Quickhull**: O(n log n) average case, but O(n²) worst case
        - **Jarvis March**: O(nh) where h is number of hull points
        - **SciPy (Qhull)**: Highly optimized O(n log n) implementation
        """)