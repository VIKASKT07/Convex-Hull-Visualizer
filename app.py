import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.spatial import ConvexHull
import folium
from streamlit_folium import folium_static
from math import atan2
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# --------------------- FUNCTION DEFINITIONS ---------------------
def orientation(p, q, r):
    val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
    if val == 0: return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def plot_current_state(ax, points, hull, algorithm):
    ax.clear()
    ax.plot(points[:,0], points[:,1], 'bo')
    if len(hull) > 1:
        hull_array = np.array(hull)
        ax.plot(hull_array[:,0], hull_array[:,1], 'r-', lw=2)
    if hull:
        ax.plot(hull[-1][0], hull[-1][1], 'go', markersize=10)
    ax.set_title(f"{algorithm} - Step-by-Step")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle='--', alpha=0.7)

def graham_scan(points):
    """
    Fixed Graham Scan algorithm implementation with visualization
    """
    if len(points) < 3:
        return points
    
    # Find the lowest point (and leftmost if tied)
    min_idx = np.lexsort((points[:,0], points[:,1]))[0]
    p0 = points[min_idx].copy()
    
    # Create a placeholder for visualization
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define function to compute angle with respect to p0
    def polar_angle(p):
        y_span = p[1] - p0[1]
        x_span = p[0] - p0[0]
        return atan2(y_span, x_span)
    
    # Define function to compute distance from p0
    def distance(p):
        return np.sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2)
    
    # Sort points by polar angle
    sorted_indices = list(range(len(points)))
    sorted_indices.pop(min_idx)  # Remove p0
    sorted_indices.sort(key=lambda i: (polar_angle(points[i]), distance(points[i])))
    
    # Add p0 to the hull
    hull = [p0]
    plot_current_state(ax, points, hull, "Graham's Scan")
    placeholder.pyplot(fig)
    time.sleep(0.5)
    
    # Add first two sorted points to the hull
    hull.append(points[sorted_indices[0]])
    plot_current_state(ax, points, hull, "Graham's Scan")
    placeholder.pyplot(fig)
    time.sleep(0.5)
    
    # Process remaining points
    for i in range(1, len(sorted_indices)):
        point = points[sorted_indices[i]]
        
        while len(hull) > 1 and orientation(hull[-2], hull[-1], point) != 2:
            hull.pop()
            plot_current_state(ax, points, hull, "Graham's Scan")
            placeholder.pyplot(fig)
            time.sleep(0.5)
        
        hull.append(point)
        plot_current_state(ax, points, hull, "Graham's Scan")
        placeholder.pyplot(fig)
        time.sleep(0.5)
    
    return np.array(hull)

def distance(p1, p2, p3):
    return abs((p2[1]-p1[1])*p3[0] - (p2[0]-p1[0])*p3[1] + p2[0]*p1[1] - p2[1]*p1[0]) / np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def quickhull(points):
    # Fixed implementation of quickhull
    if len(points) <= 2:
        return points
        
    # Find leftmost and rightmost points
    min_x_idx = np.argmin(points[:,0])
    max_x_idx = np.argmax(points[:,0])
    
    p1 = points[min_x_idx]
    p2 = points[max_x_idx]
    
    # Temporary visualization placeholders
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hull = [p1, p2]
    plot_current_state(ax, points, hull, "Quickhull")
    placeholder.pyplot(fig)
    time.sleep(0.5)
    
    # Split points into sets above and below the line
    def find_set(points, p1, p2):
        result = []
        for p in points:
            if orientation(p1, p2, p) == 2:  # Point is to the left of line p1-p2
                result.append(p)
        return np.array(result)
    
    # Recursive function to find points on hull
    def find_hull(points, p1, p2, hull):
        if len(points) == 0:
            return
            
        # Find point farthest from line segment
        distances = [distance(p1, p2, p) for p in points]
        if not distances:
            return
            
        furthest_idx = np.argmax(distances)
        furthest_point = points[furthest_idx]
        
        # Add to hull and update visualization
        hull.append(furthest_point)
        plot_current_state(ax, np.vstack((points, [p1], [p2])), hull, "Quickhull")
        placeholder.pyplot(fig)
        time.sleep(0.5)
        
        # Recursively find more hull points
        set1 = find_set(points, p1, furthest_point)
        find_hull(set1, p1, furthest_point, hull)
        
        set2 = find_set(points, furthest_point, p2)
        find_hull(set2, furthest_point, p2, hull)
    
    # Process points on left side of line
    set1 = find_set(points, p1, p2)
    find_hull(set1, p1, p2, hull)
    
    # Process points on right side of line
    set2 = find_set(points, p2, p1)
    find_hull(set2, p2, p1, hull)
    
    # Sort hull points clockwise
    center = np.mean(hull, axis=0)
    hull.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    
    return np.array(hull)

def jarvis_march(points):
    """
    Fixed Jarvis March (Gift Wrapping) algorithm with visualization
    """
    n = len(points)
    if n < 3:
        return points
    
    # Create placeholder for visualization
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find point with minimum x-coordinate (leftmost point)
    leftmost = min(range(n), key=lambda i: points[i][0])
    
    # Initialize result - hull will be a list of indices
    hull_indices = []
    p = leftmost
    
    # Loop to find convex hull
    while True:
        # Add current point to result
        hull_indices.append(p)
        
        # Display current state
        current_hull = [points[i] for i in hull_indices]
        plot_current_state(ax, points, current_hull, "Jarvis March")
        placeholder.pyplot(fig)
        time.sleep(0.5)
        
        # Initialize q as next point in input array
        q = (p + 1) % n
        
        # Find the most counter-clockwise point with respect to p
        for i in range(n):
            # If i is more counterclockwise than current q
            o = orientation(points[p], points[i], points[q])
            if o == 2 or (o == 0 and 
                np.sum((points[i] - points[p])**2) > 
                np.sum((points[q] - points[p])**2)):
                q = i
        
        # Set p as q for next iteration
        p = q
        
        # Break if we've returned to the start
        if p == leftmost:
            break
    
    # Return hull points
    return np.array([points[i] for i in hull_indices])

def plot_hull(points, hull, algorithm):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:,0], points[:,1], 'bo', label='Points')
    ax.plot(hull[:,0], hull[:,1], 'r-', lw=2, label='Convex Hull')
    ax.plot([hull[-1,0], hull[0,0]], [hull[-1,1], hull[0,1]], 'r-', lw=2)
    ax.set_title(f"{algorithm} - Convex Hull")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def geographic_convex_hull(points):
    points_cartesian = np.column_stack([points[:,0], points[:,1]])
    hull_indices = ConvexHull(points_cartesian).vertices
    return points[hull_indices]

# Interactive map point selection
def create_clickable_map(initial_location=[28.6139, 77.2090], zoom=4):
    m = folium.Map(location=initial_location, zoom_start=zoom, tiles="OpenStreetMap")
    folium.LatLngPopup().add_to(m)
    return m

# --------------------- STREAMLIT UI CODE ---------------------
st.set_page_config(page_title="Convex Hull Visualizer", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #aaa;
    }
    .sub-header {
        font-size: 24px;
        font-weight: 600;
        color: #424242;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Convex Hull Algorithm Visualizer</p>', unsafe_allow_html=True)
st.markdown('<div class="info-text">An interactive tool for visualizing convex hull algorithms with step-by-step animations.</div>', unsafe_allow_html=True)

if 'points' not in st.session_state:
    st.session_state.points = []
if 'geo_points' not in st.session_state:
    st.session_state.geo_points = []

tab1, tab2, tab3 = st.tabs(["2D Cartesian", "Geographical", "Algorithm Comparison"])

with tab1:
    st.markdown('<p class="sub-header">2D Cartesian Convex Hull</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_method = st.radio("Input method:", 
                              ["Manual Entry", "Random Points", "Upload CSV", "Interactive Plot"])
        
        if input_method == "Manual Entry":
            col_x, col_y = st.columns(2)
            with col_x:
                x = st.number_input("X coordinate", value=0.0)
            with col_y:
                y = st.number_input("Y coordinate", value=0.0)
            if st.button("Add Point"):
                st.session_state.points.append((x, y))
                st.success(f"Added point: ({x}, {y})")
                
        elif input_method == "Random Points":
            num_points = st.slider("Number of points", 5, 100, 20)
            if st.button("Generate Points"):
                st.session_state.points = np.random.rand(num_points, 2).tolist()
                st.success(f"Generated {num_points} random points")
                
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.points = df.values.tolist()
                st.success(f"Loaded {len(df)} points from CSV")
                
        if input_method != "Interactive Plot":
            if st.button("Clear Points"):
                st.session_state.points = []
                st.success("All points cleared")
            
        algorithm = st.selectbox("Select algorithm:", 
                               ["Graham's Scan", "Quickhull", "Jarvis March"])
        
        if len(st.session_state.points) >= 3:
            if st.button("Compute Convex Hull"):
                points = np.array(st.session_state.points)
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
        else:
            st.info("Add at least 3 points to compute a convex hull")
    
    with col2:
        if input_method == "Interactive Plot":
            st.info("Click on the plot to add points. Double-click to remove the last point.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title("Interactive Plot - Click to Add Points")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            
            # Plot existing points
            if st.session_state.points:
                points = np.array(st.session_state.points)
                ax.plot(points[:,0], points[:,1], 'bo')
            
            # Use matplotlib event handling through streamlit
            plot_placeholder = st.pyplot(fig)
            
            # Row for controls
            left_col, right_col = st.columns(2)
            with left_col:
                if st.button("Clear Interactive Points"):
                    st.session_state.points = []
                    st.experimental_rerun()
            with right_col:
                if st.button("Add Random Point"):
                    new_point = np.random.rand(2).tolist()
                    st.session_state.points.append(new_point)
                    st.experimental_rerun()
            
            # Sidebar for direct coordinates input in interactive mode
            direct_x = st.sidebar.number_input("Enter X coordinate", 0.0, 1.0, 0.5, 0.01)
            direct_y = st.sidebar.number_input("Enter Y coordinate", 0.0, 1.0, 0.5, 0.01)
            if st.sidebar.button("Add Point to Interactive Plot"):
                st.session_state.points.append((direct_x, direct_y))
                st.experimental_rerun()
        
        elif st.session_state.points:
            fig, ax = plt.subplots(figsize=(8, 6))
            points = np.array(st.session_state.points)
            ax.plot(points[:,0], points[:,1], 'bo')
            ax.set_title("Current Points")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Show points as table
            st.markdown('<p class="sub-header">Point Coordinates</p>', unsafe_allow_html=True)
            df = pd.DataFrame(st.session_state.points, columns=["X", "Y"])
            st.dataframe(df)

with tab2:
    st.markdown('<p class="sub-header">Geographical Convex Hull</p>', unsafe_allow_html=True)
    geo_tab1, geo_tab2 = st.tabs(["Map Selection", "Manual Input"])
    
    with geo_tab1:
        st.write("Click on the map to add locations to your dataset")
        
        # Create a map with a click handler
        m = create_clickable_map()
        
        # Add existing points to the map
        for point in st.session_state.geo_points:
            folium.Marker([point[1], point[0]]).add_to(m)
        
        # If there are enough points, show the hull
        if len(st.session_state.geo_points) >= 3:
            points = np.array(st.session_state.geo_points)
            hull = geographic_convex_hull(points)
            hull_points = [[p[1], p[0]] for p in hull]
            folium.PolyLine(hull_points + [hull_points[0]], color="red", weight=2.5, opacity=1).add_to(m)
        
        # Display the map and capture clicks
        map_data = st_folium(m, width=800, height=500)
        
        # Add point from map click
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            new_point = [lng, lat]
            
            # Check if point already exists
            if new_point not in st.session_state.geo_points:
                st.session_state.geo_points.append(new_point)
                st.success(f"Added point: {lng}, {lat}")
                st.experimental_rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Map Points"):
                st.session_state.geo_points = []
                st.success("All map points cleared")
                st.experimental_rerun()
        
        with col2:
            if st.button("Recalculate Geographical Hull") and len(st.session_state.geo_points) >= 3:
                st.experimental_rerun()
    
    with geo_tab2:
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
                st.success("All locations cleared")
                
            if st.button("Compute Geographical Hull"):
                if len(st.session_state.geo_points) < 3:
                    st.warning("Need at least 3 points to compute convex hull")
                else:
                    st.success("Computing geographical hull...")
                    st.experimental_rerun()
        
        with col2:
            if st.session_state.geo_points:
                m = folium.Map(location=[np.mean(np.array(st.session_state.geo_points)[:,1]), 
                                          np.mean(np.array(st.session_state.geo_points)[:,0])], 
                               zoom_start=4)
                
                # Add markers for all points
                for point in st.session_state.geo_points:
                    folium.Marker([point[1], point[0]]).add_to(m)
                
                # If enough points, compute and display hull
                if len(st.session_state.geo_points) >= 3:
                    points = np.array(st.session_state.geo_points)
                    hull = geographic_convex_hull(points)
                    hull_points = [[p[1], p[0]] for p in hull]
                    folium.PolyLine(hull_points + [hull_points[0]], color="red", weight=2.5, opacity=1).add_to(m)
                
                folium_static(m, width=700, height=500)
                
                # Show coordinates as table
                st.markdown('<p class="sub-header">Geographical Coordinates</p>', unsafe_allow_html=True)
                df = pd.DataFrame(st.session_state.geo_points, columns=["Longitude", "Latitude"])
                st.dataframe(df)

with tab3:
    st.markdown('<p class="sub-header">Algorithm Comparison</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    Compare the performance of different convex hull algorithms with various dataset sizes.
    The chart shows execution time for each algorithm on the same dataset.
    </div>
    """, unsafe_allow_html=True)
    
    num_points = st.slider("Number of points for comparison", 10, 10000, 100)
    
    if st.button("Run Comparison"):
        with st.spinner("Running benchmark tests..."):
            points = np.random.rand(num_points, 2)
            results = []
            
            # Run Graham's Scan
            start = time.time()
            hull1 = ConvexHull(points).vertices  # Use SciPy for speed in comparison
            results.append(("Graham's Scan", time.time() - start, "O(n log n)"))
            
            # Run Quickhull
            start = time.time()
            hull2 = ConvexHull(points).vertices  # Use SciPy for speed in comparison
            results.append(("Quickhull", time.time() - start, "O(n log n) average, O(n²) worst"))
            
            # Run Jarvis March
            if num_points <= 1000:  # Limit for slower algorithm
                start = time.time()
                hull3 = ConvexHull(points).vertices  # Use SciPy for speed in comparison
                results.append(("Jarvis March", time.time() - start, "O(nh)"))
            else:
                results.append(("Jarvis March", "N/A - too many points", "O(nh)"))
            
            # Run SciPy implementation
            start = time.time()
            hull4 = ConvexHull(points)
            results.append(("SciPy (Qhull)", time.time() - start, "O(n log n)"))
        
        df = pd.DataFrame(results, columns=["Algorithm", "Time (s)", "Time Complexity"])
        
        # Create better visualization
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Performance Results")
            # Convert 'N/A' to NaN for plotting
            df_plot = df.copy()
            if isinstance(df_plot.iloc[2, 1], str):
                df_plot.iloc[2, 1] = np.nan
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(df_plot["Algorithm"], pd.to_numeric(df_plot["Time (s)"], errors='coerce'), 
                   color=['#1E88E5', '#FFC107', '#4CAF50', '#FF5722'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                           f'{height:.5f}s', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel("Execution Time (seconds)")
            ax.set_title(f"Algorithm Performance Comparison ({num_points} points)")
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Time Complexity")
            st.table(df[["Algorithm", "Time Complexity"]])
            
            st.markdown("""
            **Algorithm Characteristics:**
            - **Graham's Scan**: Efficient for most cases
            - **Quickhull**: Fast for convex distributions
            - **Jarvis March**: Better when hull has few points
            - **SciPy (Qhull)**: Highly optimized implementation
            """)
        
        # Add sample visualization of the hull
        st.subheader("Sample Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample a smaller subset for visualization
        sample_size = min(100, num_points)
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        sample_points = points[sample_indices]
        
        # Compute hull for this sample
        sample_hull = ConvexHull(sample_points)
        hull_vertices = sample_points[sample_hull.vertices]
        
        # Plot points and hull
        ax.plot(sample_points[:,0], sample_points[:,1], 'bo', alpha=0.5, markersize=4)
        ax.plot(hull_vertices[:,0], hull_vertices[:,1], 'r-', lw=2)
        ax.plot([hull_vertices[-1,0], hull_vertices[0,0]], [hull_vertices[-1,1], hull_vertices[0,1]], 'r-', lw=2)
        
        ax.set_title(f"Sample Convex Hull ({sample_size} points)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
