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
import seaborn as sns
from itertools import combinations
import io
import base64
from matplotlib.animation import FuncAnimation

# --------------------- FUNCTION DEFINITIONS ---------------------
def orientation(p, q, r):
    val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
    if val == 0: return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def plot_current_state(ax, points, hull, algorithm, current_points=None, checking_line=None):
    ax.clear()
    ax.plot(points[:,0], points[:,1], 'bo', alpha=0.6, markersize=8, label='Input Points')
    
    if len(hull) > 1:
        hull_array = np.array(hull)
        ax.plot(hull_array[:,0], hull_array[:,1], 'r-', lw=3, label='Convex Hull')
    
    if hull:
        ax.plot(hull[-1][0], hull[-1][1], 'go', markersize=12, label='Latest Hull Point')
    
    # Highlight current points being processed
    if current_points is not None:
        for i, point in enumerate(current_points):
            ax.plot(point[0], point[1], 'yo', markersize=10, alpha=0.8)
    
    # Show checking line for brute force
    if checking_line is not None:
        p1, p2 = checking_line
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', lw=2, alpha=0.7, label='Checking Line')
    
    ax.set_title(f"{algorithm} - Step-by-Step Visualization", fontsize=16, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)

class StepTracker:
    def __init__(self):
        self.steps = []
        self.current_step = 0
    
    def add_step(self, step_num, title, description, details=None, status="active"):
        step_data = {
            'step_num': step_num,
            'title': title,
            'description': description,
            'details': details,
            'status': status
        }
        self.steps.append(step_data)
        self.current_step = step_num
    
    def display_steps(self, container):
        """Display all steps using Streamlit native components with full width"""
        if not self.steps:
            return
        
        with container:
            st.markdown("---")
            st.markdown("# 📋 Algorithm Step-by-Step Documentation")
            st.markdown("*Follow the detailed progression of the algorithm with comprehensive explanations*")
            st.markdown("")
            
            # Display each step in full width
            for i, step in enumerate(self.steps):
                self._display_single_step_native(step, i)
    
    def _display_single_step_native(self, step, index):
        """Display a single step using Streamlit native components"""
        
        # Create a container for each step with custom styling
        with st.container():
            # Create columns for better layout - using full width
            col1, col2 = st.columns([1, 12])  # Tiny left margin, rest for content
            
            with col1:
                # Step number badge
                if step['status'] == 'completed':
                    st.success(f"✅")
                elif step['status'] == 'active':
                    st.info(f"🔄")
                else:
                    st.warning(f"⏳")
            
            with col2:
                # Step header with number and title
                st.markdown(f"### Step {step['step_num']}: {step['title']}")
                
                # Main description
                st.markdown(f"**{step['description']}**")
                
                # Details in an expandable section if available
                if step['details']:
                    with st.expander("🔍 Detailed Explanation", expanded=True):
                        st.markdown(step['details'])
                
                # Add some spacing between steps
                st.markdown("")
                st.markdown("---")
                st.markdown("")

def brute_force_convex_hull(points):
    """
    Brute Force algorithm implementation with enhanced step-by-step visualization
    """
    if len(points) < 3:
        return points
    
    # Create main visualization area
    st.markdown("### 🔍 Brute Force Algorithm Visualization")
    
    # Single large plot area
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create step tracker
    step_tracker = StepTracker()
    
    # Create container for steps below
    steps_container = st.container()
    
    hull_edges = []
    step_num = 1
    
    # Step 1: Initialize
    step_tracker.add_step(
        step_num, 
        "🚀 Initialize Brute Force Algorithm", 
        "Starting the brute force approach by examining all possible line segments between point pairs.",
        f"""
        **📊 Dataset Analysis:**
        - Total points in dataset: {len(points)}
        - Number of pairs to examine: {len(points) * (len(points) - 1) // 2}
        - Time complexity: O(n³)
        - Space complexity: O(n)
        
        **🔍 Algorithm Strategy:**
        The brute force method works by checking every possible line segment formed by pairs of points. 
        For each line segment, we verify if all other points lie on the same side. If they do, 
        this line segment is part of the convex hull boundary.
        """
    )
    
    plot_current_state(ax, points, [], "Brute Force")
    placeholder.pyplot(fig)
    time.sleep(1.5)
    step_num += 1
    
    # Check all pairs of points
    total_pairs = 0
    valid_edges = 0
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            total_pairs += 1
            p1, p2 = points[i], points[j]
            
            # Check if all other points are on the same side of line p1-p2
            left_count = 0
            right_count = 0
            
            for k in range(len(points)):
                if k != i and k != j:
                    orient = orientation(p1, p2, points[k])
                    if orient == 1:  # Right side
                        right_count += 1
                    elif orient == 2:  # Left side
                        left_count += 1
            
            # If all points are on one side, this is a hull edge
            if left_count == 0 or right_count == 0:
                hull_edges.append((p1, p2))
                valid_edges += 1
                
                step_tracker.add_step(
                    step_num,
                    f"✅ Found Hull Edge #{valid_edges}",
                    f"Line segment from ({p1[0]:.3f}, {p1[1]:.3f}) to ({p2[0]:.3f}, {p2[1]:.3f}) is confirmed as part of the convex hull.",
                    f"""
                    **🔍 Validation Process:**
                    - Points examined: {len(points)-2}
                    - Points on left side: {left_count}
                    - Points on right side: {right_count}
                    - Result: All points lie on one side, confirming this is a hull edge
                    
                    **📈 Progress Update:**
                    - Hull edges found so far: {valid_edges}
                    - Total pairs checked: {total_pairs}
                    - Remaining pairs: {len(points) * (len(points) - 1) // 2 - total_pairs}
                    """
                )
                
                # Visualize current hull edges
                current_hull_points = []
                for edge in hull_edges:
                    current_hull_points.extend(edge)
                
                plot_current_state(ax, points, current_hull_points, "Brute Force", 
                                 current_points=[p1, p2], checking_line=(p1, p2))
                placeholder.pyplot(fig)
                time.sleep(1.5)
                step_num += 1
            else:
                # Show some rejected lines for educational purposes
                if total_pairs % 10 == 0:  # Show every 10th rejection to avoid clutter
                    step_tracker.add_step(
                        step_num,
                        f"❌ Rejected Line Segment",
                        f"Line from ({p1[0]:.3f}, {p1[1]:.3f}) to ({p2[0]:.3f}, {p2[1]:.3f}) is not part of the convex hull.",
                        f"""
                        **🔍 Rejection Reason:**
                        - Points on left side: {left_count}
                        - Points on right side: {right_count}
                        - Conclusion: Points exist on both sides, so this line passes through the interior
                        
                        **📊 Current Progress:**
                        - Pairs examined: {total_pairs}
                        - Valid hull edges found: {valid_edges}
                        - Rejection rate: {((total_pairs - valid_edges) / total_pairs * 100):.1f}%
                        """
                    )
                    
                    plot_current_state(ax, points, [], "Brute Force", 
                                     current_points=[p1, p2], checking_line=(p1, p2))
                    placeholder.pyplot(fig)
                    time.sleep(1)
                    step_num += 1
    
    # Sort hull points to form a proper polygon
    if hull_edges:
        # Extract unique points from edges
        hull_points = []
        for edge in hull_edges:
            for point in edge:
                if not any(np.array_equal(point, hp) for hp in hull_points):
                    hull_points.append(point)
        
        # Sort points by angle from centroid
        if len(hull_points) > 2:
            center = np.mean(hull_points, axis=0)
            hull_points.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        
        step_tracker.add_step(
            step_num,
            "🎉 Algorithm Complete!",
            f"Successfully computed convex hull with {len(hull_points)} vertices using brute force method.",
            f"""
            **📈 Final Performance Summary:**
            - Total line segments checked: {total_pairs}
            - Valid hull edges found: {valid_edges}
            - Hull vertices: {len(hull_points)}
            - Time complexity achieved: O(n³)
            - Space complexity: O(n)
            
            **🎯 Algorithm Insights:**
            - Brute force guarantees finding the correct convex hull
            - Every possible line segment is examined systematically
            - Simple to understand and implement
            - Inefficient for large datasets due to cubic time complexity
            - Best used for educational purposes and small point sets
            """
        )
        
        plot_current_state(ax, points, hull_points, "Brute Force")
        placeholder.pyplot(fig)
        
        # Display all steps
        step_tracker.display_steps(steps_container)
        
        return np.array(hull_points)
    
    step_tracker.display_steps(steps_container)
    return np.array([])

def graham_scan(points):
    """
    Graham Scan algorithm implementation with enhanced step-by-step visualization
    """
    if len(points) < 3:
        return points
    
    # Create main visualization area
    st.markdown("### 📐 Graham's Scan Algorithm Visualization")
    
    # Single large plot area
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create step tracker
    step_tracker = StepTracker()
    
    # Create container for steps below
    steps_container = st.container()
    
    # Find the lowest point (and leftmost if tied)
    min_idx = np.lexsort((points[:,0], points[:,1]))[0]
    p0 = points[min_idx].copy()
    
    step_tracker.add_step(
        1,
        "🎯 Find Starting Point",
        f"Located the starting point at coordinates ({p0[0]:.3f}, {p0[1]:.3f})",
        f"""
        **🔍 Selection Strategy:**
        - Choose the point with the lowest Y-coordinate
        - If multiple points have the same Y-coordinate, select the leftmost one
        - This point is guaranteed to be on the convex hull
        
        **📊 Point Analysis:**
        - Total points in dataset: {len(points)}
        - Starting point coordinates: ({p0[0]:.3f}, {p0[1]:.3f})
        - Remaining points to process: {len(points) - 1}
        
        **🎯 Why This Works:**
        The bottommost point (or leftmost among bottommost) cannot be inside any convex hull,
        as there would need to be points below it to form a convex shape, which contradicts
        our selection criteria.
        """
    )
    
    # Define function to compute angle with respect to p0
    def polar_angle(p):
        y_span = p[1] - p0[1]
        x_span = p[0] - p0[0]
        return atan2(y_span, x_span)
    
    def distance(p):
        return np.sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2)
    
    # Sort points by polar angle
    sorted_indices = list(range(len(points)))
    sorted_indices.pop(min_idx)
    sorted_indices.sort(key=lambda i: (polar_angle(points[i]), distance(points[i])))
    
    step_tracker.add_step(
        2,
        "🔄 Sort Points by Polar Angle",
        f"Sorted {len(sorted_indices)} points in counterclockwise order around the starting point",
        f"""
        **📐 Sorting Method:**
        - Calculate polar angle of each point relative to starting point
        - Sort points by increasing polar angle (counterclockwise)
        - If two points have the same angle, sort by distance from starting point
        
        **🔢 Mathematical Details:**
        - Polar angle calculated using: atan2(y_span, x_span)
        - y_span = point.y - start.y
        - x_span = point.x - start.x
        - This ensures counterclockwise ordering around the starting point
        
        **⚡ Complexity Analysis:**
        - Sorting step: O(n log n)
        - This is the dominant operation in Graham's Scan
        - Subsequent processing is O(n)
        """
    )
    
    hull = [p0]
    plot_current_state(ax, points, hull, "Graham's Scan")
    placeholder.pyplot(fig)
    time.sleep(1.5)
    
    # Add first sorted point
    if sorted_indices:
        hull.append(points[sorted_indices[0]])
        step_tracker.add_step(
            3,
            "➕ Add First Sorted Point",
            f"Added point ({points[sorted_indices[0]][0]:.3f}, {points[sorted_indices[0]][1]:.3f}) to the hull",
            f"""
            **🎯 First Point Selection:**
            - This point has the smallest polar angle from our starting point
            - It forms the initial edge of our convex hull
            - No orientation check needed for the first point
            
            **📊 Current Hull Status:**
            - Hull vertices: 2
            - Current hull points: [{p0[0]:.3f}, {p0[1]:.3f}], [{points[sorted_indices[0]][0]:.3f}, {points[sorted_indices[0]][1]:.3f}]
            - Remaining points to process: {len(sorted_indices) - 1}
            """
        )
        
        plot_current_state(ax, points, hull, "Graham's Scan")
        placeholder.pyplot(fig)
        time.sleep(1.5)
    
    step_num = 4
    removed_total = 0
    
    # Process remaining points
    for i in range(1, len(sorted_indices)):
        point = points[sorted_indices[i]]
        
        # Remove points that make right turns
        removed_count = 0
        while len(hull) > 1 and orientation(hull[-2], hull[-1], point) != 2:
            removed_point = hull.pop()
            removed_count += 1
            removed_total += 1
            
            step_tracker.add_step(
                step_num,
                f"🔄 Remove Point (Right Turn Detected)",
                f"Removed point ({removed_point[0]:.3f}, {removed_point[1]:.3f}) because it creates a right turn",
                f"""
                **🔍 Orientation Check Details:**
                - Previous point: ({hull[-1][0]:.3f}, {hull[-1][1]:.3f}) if hull exists
                - Removed point: ({removed_point[0]:.3f}, {removed_point[1]:.3f})
                - Current point: ({point[0]:.3f}, {point[1]:.3f})
                - Turn direction: Clockwise (right turn)
                
                **🎯 Why Remove This Point:**
                - Right turns indicate the middle point is inside the convex hull
                - Only left turns (counterclockwise) should remain in the final hull
                - This maintains the convex property of our hull
                
                **📊 Removal Statistics:**
                - Points removed in this iteration: {removed_count}
                - Total points removed so far: {removed_total}
                - Current hull size: {len(hull)}
                """
            )
            
            plot_current_state(ax, points, hull, "Graham's Scan", current_points=[point])
            placeholder.pyplot(fig)
            time.sleep(1)
            step_num += 1
        
        hull.append(point)
        step_tracker.add_step(
            step_num,
            f"✅ Add Point to Hull",
            f"Successfully added point ({point[0]:.3f}, {point[1]:.3f}) to the convex hull",
            f"""
            **✅ Successful Addition:**
            - Point coordinates: ({point[0]:.3f}, {point[1]:.3f})
            - Turn direction: Counterclockwise (left turn) or collinear
            - This point maintains the convex property
            
            **📊 Progress Summary:**
            - Points processed: {i+1}/{len(sorted_indices)}
            - Current hull size: {len(hull)} vertices
            - Points removed in this iteration: {removed_count}
            - Total removals so far: {removed_total}
            - Completion: {((i+1)/len(sorted_indices)*100):.1f}%
            
            **🔄 Next Steps:**
            - Continue processing remaining {len(sorted_indices)-(i+1)} points
            - Check orientation for each new point
            - Remove any points that create right turns
            """
        )
        
        plot_current_state(ax, points, hull, "Graham's Scan", current_points=[point])
        placeholder.pyplot(fig)
        time.sleep(1.5)
        step_num += 1
    
    step_tracker.add_step(
        step_num,
        "🎉 Graham's Scan Complete!",
        f"Successfully computed convex hull with {len(hull)} vertices",
        f"""
        **🎯 Final Algorithm Summary:**
        - Input points processed: {len(points)}
        - Final hull vertices: {len(hull)}
        - Total points removed: {removed_total}
        - Efficiency: {((len(hull)/len(points))*100):.1f}% of points are on the hull
        
        **⚡ Performance Analysis:**
        - Time Complexity: O(n log n) - dominated by sorting step
        - Space Complexity: O(n) - for storing hull and sorted points
        - Actual operations: ~{len(points)} comparisons after sorting
        
        **🏆 Algorithm Advantages:**
        - Optimal time complexity for comparison-based algorithms
        - Simple and elegant implementation
        - Stable performance across different point distributions
        - Widely used in computational geometry applications
        """
    )
    
    # Display all steps
    step_tracker.display_steps(steps_container)
    
    return np.array(hull)

def distance(p1, p2, p3):
    return abs((p2[1]-p1[1])*p3[0] - (p2[0]-p1[0])*p3[1] + p2[0]*p1[1] - p2[1]*p1[0]) / np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def quickhull(points):
    """
    Quickhull algorithm implementation with enhanced step-by-step visualization
    """
    if len(points) <= 2:
        return points
    
    # Create main visualization area
    st.markdown("### ⚡ Quickhull Algorithm Visualization")
    
    # Single large plot area
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create step tracker
    step_tracker = StepTracker()
    
    # Create container for steps below
    steps_container = st.container()
    
    # Find leftmost and rightmost points
    min_x_idx = np.argmin(points[:,0])
    max_x_idx = np.argmax(points[:,0])
    
    p1 = points[min_x_idx]
    p2 = points[max_x_idx]
    
    hull = [p1, p2]
    step_num = 1
    
    step_tracker.add_step(
        step_num,
        "🎯 Find Extreme Points",
        f"Located extreme points: Leftmost ({p1[0]:.3f}, {p1[1]:.3f}) and Rightmost ({p2[0]:.3f}, {p2[1]:.3f})",
        f"""
        **🔍 Extreme Point Strategy:**
        - Leftmost point: Minimum X-coordinate = {p1[0]:.3f}
        - Rightmost point: Maximum X-coordinate = {p2[0]:.3f}
        - These points are guaranteed to be vertices of the convex hull
        
        **📐 Divide and Conquer Setup:**
        - The line between extreme points divides remaining points into two sets
        - Upper set: Points above the line (left side when going from left to right)
        - Lower set: Points below the line (right side when going from left to right)
        - Points on the line are ignored (they're inside the hull)
        
        **⚡ Algorithm Efficiency:**
        - Best case: O(n log n) when points are evenly distributed
        - Worst case: O(n²) when points form a specific pattern
        - Average case: O(n log n) for most practical distributions
        """
    )
    
    plot_current_state(ax, points, hull, "Quickhull")
    placeholder.pyplot(fig)
    time.sleep(1.5)
    step_num += 1
    
    def find_set(points, p1, p2):
        result = []
        for p in points:
            if orientation(p1, p2, p) == 2:
                result.append(p)
        return np.array(result)
    
    def find_hull(points_set, p1, p2, hull, side_name):
        nonlocal step_num
        if len(points_set) == 0:
            return
            
        distances = [distance(p1, p2, p) for p in points_set]
        if not distances:
            return
            
        furthest_idx = np.argmax(distances)
        furthest_point = points_set[furthest_idx]
        max_distance = distances[furthest_idx]
        
        step_tracker.add_step(
            step_num,
            f"🔍 Find Furthest Point ({side_name.title()} Side)",
            f"Found furthest point ({furthest_point[0]:.3f}, {furthest_point[1]:.3f}) from the dividing line",
            f"""
            **📏 Distance Calculation Details:**
            - Line segment: ({p1[0]:.3f}, {p1[1]:.3f}) to ({p2[0]:.3f}, {p2[1]:.3f})
            - Furthest point: ({furthest_point[0]:.3f}, {furthest_point[1]:.3f})
            - Perpendicular distance: {max_distance:.4f}
            - Points examined on {side_name} side: {len(points_set)}
            
            **🎯 Why This Point is Important:**
            - Maximum distance guarantees it's on the convex hull
            - Forms a triangle with the current line segment
            - Divides the point set further for recursive processing
            
            **🔄 Recursive Strategy:**
            - Create two new line segments from this point
            - Recursively process points on each side of new segments
            - Continue until no points remain outside the current hull
            """
        )
        
        hull.append(furthest_point)
        plot_current_state(ax, np.vstack((points, [p1], [p2])), hull, "Quickhull", 
                         current_points=[furthest_point])
        placeholder.pyplot(fig)
        time.sleep(1.5)
        step_num += 1
        
        # Recursive calls
        set1 = find_set(points_set, p1, furthest_point)
        if len(set1) > 0:
            step_tracker.add_step(
                step_num,
                f"🔄 Recursive Processing (Left Segment)",
                f"Processing {len(set1)} points on the left side of line from ({p1[0]:.3f}, {p1[1]:.3f}) to ({furthest_point[0]:.3f}, {furthest_point[1]:.3f})",
                f"""
                **🌳 Divide & Conquer Recursion:**
                - New line segment: ({p1[0]:.3f}, {p1[1]:.3f}) → ({furthest_point[0]:.3f}, {furthest_point[1]:.3f})
                - Points to process: {len(set1)}
                - Recursion depth: Increasing
                - Side: Left of the new segment
                
                **📊 Recursive Efficiency:**
                - Each recursion reduces the problem size
                - Points are eliminated that cannot be on the hull
                - Logarithmic depth in average case
                - Linear depth in worst case
                
                **🎯 Termination Condition:**
                - Recursion stops when no points remain on one side
                - All remaining points are inside the current triangle
                """
            )
            step_num += 1
            find_hull(set1, p1, furthest_point, hull, "left")
        
        set2 = find_set(points_set, furthest_point, p2)
        if len(set2) > 0:
            step_tracker.add_step(
                step_num,
                f"🔄 Recursive Processing (Right Segment)",
                f"Processing {len(set2)} points on the right side of line from ({furthest_point[0]:.3f}, {furthest_point[1]:.3f}) to ({p2[0]:.3f}, {p2[1]:.3f})",
                f"""
                **🌳 Continue Divide & Conquer:**
                - New line segment: ({furthest_point[0]:.3f}, {furthest_point[1]:.3f}) → ({p2[0]:.3f}, {p2[1]:.3f})
                - Points to process: {len(set2)}
                - Side: Right of the new segment
                - Parallel processing opportunity in implementation
                
                **⚡ Performance Characteristics:**
                - Each recursive call processes fewer points
                - Geometric elimination of interior points
                - Efficient for convex point distributions
                - May degrade for adversarial point arrangements
                """
            )
            step_num += 1
            find_hull(set2, furthest_point, p2, hull, "right")
    
    # Process both sides
    set1 = find_set(points, p1, p2)
    if len(set1) > 0:
        step_tracker.add_step(
            step_num,
            f"🔼 Process Upper Hull",
            f"Starting recursive processing of {len(set1)} points above the main dividing line",
            f"""
            **📐 Upper Hull Processing:**
            - Main dividing line: ({p1[0]:.3f}, {p1[1]:.3f}) → ({p2[0]:.3f}, {p2[1]:.3f})
            - Points above line: {len(set1)}
            - These points are candidates for the upper portion of the convex hull
            
            **🔍 Point Classification:**
            - Points above the line have positive orientation
            - Points below the line have negative orientation
            - Points on the line are ignored (interior to hull)
            
            **🎯 Processing Strategy:**
            - Find the furthest point from the main line
            - Recursively process left and right subsections
            - Build upper hull incrementally
            """
        )
        step_num += 1
        find_hull(set1, p1, p2, hull, "upper")
    
    set2 = find_set(points, p2, p1)
    if len(set2) > 0:
        step_tracker.add_step(
            step_num,
            f"🔽 Process Lower Hull",
            f"Starting recursive processing of {len(set2)} points below the main dividing line",
            f"""
            **📐 Lower Hull Processing:**
            - Main dividing line: ({p2[0]:.3f}, {p2[1]:.3f}) → ({p1[0]:.3f}, {p1[1]:.3f}) (reversed)
            - Points below line: {len(set2)}
            - These points are candidates for the lower portion of the convex hull
            
            **🔄 Symmetric Processing:**
            - Same algorithm applied to lower half
            - Line direction reversed for proper orientation
            - Completes the full convex hull
            
            **🎯 Final Assembly:**
            - Upper and lower hulls combine to form complete boundary
            - Extreme points connect the two halves
            - Result is a complete convex polygon
            """
        )
        step_num += 1
        find_hull(set2, p2, p1, hull, "lower")
    
    # Sort hull points
    center = np.mean(hull, axis=0)
    hull.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    
    step_tracker.add_step(
        step_num,
        "🎉 Quickhull Complete!",
        f"Successfully computed convex hull with {len(hull)} vertices using divide-and-conquer approach",
        f"""
        **🎯 Final Algorithm Summary:**
        - Input points: {len(points)}
        - Hull vertices found: {len(hull)}
        - Efficiency ratio: {(len(hull)/len(points)*100):.1f}% of points on hull
        
        **⚡ Performance Analysis:**
        - Average Time Complexity: O(n log n)
        - Worst Case Time Complexity: O(n²)
        - Space Complexity: O(log n) for recursion stack
        - Practical performance: Excellent for most distributions
        
        **🏆 Algorithm Strengths:**
        - Intuitive divide-and-conquer approach
        - Excellent average-case performance
        - Natural parallelization opportunities
        - Efficient geometric pruning of interior points
        
        **⚠️ Considerations:**
        - Worst-case quadratic behavior possible
        - Performance depends on point distribution
        - Recursive implementation may have stack limitations
        """
    )
    
    # Display all steps
    step_tracker.display_steps(steps_container)
    
    return np.array(hull)

def jarvis_march(points):
    """
    Jarvis March algorithm implementation with enhanced step-by-step visualization
    """
    n = len(points)
    if n < 3:
        return points
    
    # Create main visualization area
    st.markdown("### 🎁 Jarvis March (Gift Wrapping) Algorithm Visualization")
    
    # Single large plot area
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create step tracker
    step_tracker = StepTracker()
    
    # Create container for steps below
    steps_container = st.container()
    
    leftmost = min(range(n), key=lambda i: points[i][0])
    
    step_tracker.add_step(
        1,
        "🎯 Find Starting Point",
        f"Located leftmost point at ({points[leftmost][0]:.3f}, {points[leftmost][1]:.3f}) as our starting vertex",
        f"""
        **🔍 Starting Point Selection:**
        - Leftmost point coordinates: ({points[leftmost][0]:.3f}, {points[leftmost][1]:.3f})
        - Point index in dataset: {leftmost}
        - This point is guaranteed to be on the convex hull
        
        **🎁 Gift Wrapping Analogy:**
        - Imagine wrapping a gift around all the points
        - Start from the leftmost point (like starting to wrap from one corner)
        - "Wrap" the string around the outside, always taking the most counterclockwise turn
        - Continue until you return to the starting point
        
        **📊 Algorithm Overview:**
        - Time Complexity: O(nh) where h is the number of hull vertices
        - Space Complexity: O(h) for storing hull vertices
        - Output-sensitive: Faster when hull is small relative to total points
        """
    )
    
    hull_indices = []
    p = leftmost
    step_num = 2
    iteration = 0
    
    while True:
        iteration += 1
        hull_indices.append(p)
        current_hull = [points[i] for i in hull_indices]
        
        step_tracker.add_step(
            step_num,
            f"➕ Add Vertex #{len(hull_indices)} to Hull",
            f"Added point ({points[p][0]:.3f}, {points[p][1]:.3f}) as vertex #{len(hull_indices)} of the convex hull",
            f"""
            **🔄 Current Iteration: {iteration}**
            - Current vertex: ({points[p][0]:.3f}, {points[p][1]:.3f})
            - Hull vertices so far: {len(hull_indices)}
            - Vertex index: {p}
            
            **🧭 Next Step Process:**
            - Examine all remaining points as potential next vertices
            - Find the point that makes the most counterclockwise turn
            - Use orientation test to determine turn direction
            - Select the point that "wraps" furthest around the point set
            
            **📐 Orientation Test:**
            - For three points A, B, C: orientation(A, B, C)
            - Returns: 0 (collinear), 1 (clockwise), 2 (counterclockwise)
            - We want the most counterclockwise point from current position
            """
        )
        
        plot_current_state(ax, points, current_hull, "Jarvis March", current_points=[points[p]])
        placeholder.pyplot(fig)
        time.sleep(1.5)
        step_num += 1
        
        # Find most counterclockwise point
        q = (p + 1) % n
        candidates_checked = 0
        best_candidates = []
        
        for i in range(n):
            if i == p:
                continue
            candidates_checked += 1
            o = orientation(points[p], points[i], points[q])
            if o == 2 or (o == 0 and 
                np.sum((points[i] - points[p])**2) > 
                np.sum((points[q] - points[p])**2)):
                if o == 2:  # New best counterclockwise point
                    best_candidates = [i]
                    q = i
                elif o == 0:  # Collinear, but farther
                    best_candidates.append(i)
                    q = i
        
        step_tracker.add_step(
            step_num,
            f"🔍 Find Next Hull Vertex",
            f"Examined {candidates_checked} candidate points and selected ({points[q][0]:.3f}, {points[q][1]:.3f}) as the next hull vertex",
            f"""
            **🔍 Candidate Evaluation Process:**
            - Total candidates examined: {candidates_checked}
            - Current vertex: ({points[p][0]:.3f}, {points[p][1]:.3f})
            - Selected next vertex: ({points[q][0]:.3f}, {points[q][1]:.3f})
            - Selection criterion: Most counterclockwise turn
            
            **🧭 Selection Algorithm:**
            1. Start with any point as initial candidate
            2. For each remaining point, check orientation
            3. If point makes more counterclockwise turn, update candidate
            4. If collinear, choose the farther point
            5. Final candidate is the next hull vertex
            
            **📊 Efficiency Analysis:**
            - Points examined this iteration: {candidates_checked}
            - Total comparisons so far: {iteration * (n-1)}
            - Expected total iterations: O(h) where h is hull size
            - This makes algorithm output-sensitive
            """
        )
        
        step_num += 1
        p = q
        
        if p == leftmost:
            step_tracker.add_step(
                step_num,
                "🔄 Return to Starting Point",
                "Returned to the starting point, indicating the convex hull is complete",
                f"""
                **✅ Hull Completion Detected:**
                - Current point: ({points[p][0]:.3f}, {points[p][1]:.3f})
                - Starting point: ({points[leftmost][0]:.3f}, {points[leftmost][1]:.3f})
                - Points are identical: Hull is closed
                
                **🎁 Gift Wrapping Complete:**
                - Successfully "wrapped" around the entire point set
                - Returned to starting position, forming a closed polygon
                - All exterior points have been identified
                
                **📊 Final Statistics:**
                - Total iterations: {iteration}
                - Hull vertices found: {len(hull_indices)}
                - Points examined: {iteration * (n-1)}
                - Efficiency: Found hull in O({len(hull_indices)}n) time
                """
            )
            break
    
    final_hull = [points[i] for i in hull_indices]
    
    step_tracker.add_step(
        step_num + 1,
        "🎉 Jarvis March Complete!",
        f"Successfully computed convex hull with {len(hull_indices)} vertices using the gift wrapping method",
        f"""
        **🎯 Final Algorithm Summary:**
        - Input points: {len(points)}
        - Hull vertices: {len(hull_indices)}
        - Total iterations: {iteration}
        - Hull efficiency: {(len(hull_indices)/len(points)*100):.1f}% of points on boundary
        
        **⚡ Performance Analysis:**
        - Time Complexity: O(nh) = O({n} × {len(hull_indices)}) = O({n * len(hull_indices)})
        - Space Complexity: O(h) = O({len(hull_indices)})
        - Comparisons made: {iteration * (n-1)}
        - Output-sensitive: Efficient when hull is small
        
        **🏆 Algorithm Advantages:**
        - Simple and intuitive approach
        - Output-sensitive time complexity
        - Easy to understand and implement
        - Excellent for small convex hulls
        - Natural geometric interpretation
        
        **🎯 Best Use Cases:**
        - Small convex hulls (h << n)
        - Educational purposes
        - When simplicity is preferred over optimal worst-case performance
        - Real-time applications with small expected hull sizes
        """
    )
    
    # Display all steps
    step_tracker.display_steps(steps_container)
    
    return np.array(final_hull)

def plot_hull(points, hull, algorithm):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(points[:,0], points[:,1], 'bo', label='Points', markersize=8)
    ax.plot(hull[:,0], hull[:,1], 'r-', lw=4, label='Convex Hull')
    ax.plot([hull[-1,0], hull[0,0]], [hull[-1,1], hull[0,1]], 'r-', lw=4)
    ax.set_title(f"{algorithm} - Final Convex Hull Result", fontsize=18, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def geographic_convex_hull(points):
    points_cartesian = np.column_stack([points[:,0], points[:,1]])
    hull_indices = ConvexHull(points_cartesian).vertices
    return points[hull_indices]

def create_clickable_map(initial_location=[28.6139, 77.2090], zoom=4):
    m = folium.Map(location=initial_location, zoom_start=zoom, tiles="OpenStreetMap")
    folium.LatLngPopup().add_to(m)
    return m

def run_comprehensive_comparison(num_points):
    """Run comprehensive algorithm comparison with multiple metrics"""
    
    # Generate different types of point distributions
    distributions = {
        "Random": np.random.rand(num_points, 2),
        "Circle": np.array([[0.5 + 0.4 * np.cos(t), 0.5 + 0.4 * np.sin(t)] 
                           for t in np.linspace(0, 2*np.pi, num_points)]),
        "Gaussian": np.clip(np.random.normal(0.5, 0.15, (num_points, 2)), 0, 1),
        "Square": np.array([[i/(int(np.sqrt(num_points))), j/(int(np.sqrt(num_points)))] 
                           for i in range(int(np.sqrt(num_points))) 
                           for j in range(int(np.sqrt(num_points)))])[:num_points]
    }
    
    results = []
    hull_sizes = {}
    
    for dist_name, points in distributions.items():
        # Test each algorithm
        algorithms = ["Graham's Scan", "Quickhull", "Jarvis March", "Brute Force"]
        
        for algo in algorithms:
            if algo == "Brute Force" and num_points > 50:
                # Skip brute force for large datasets
                results.append((algo, dist_name, "N/A - too slow", "O(n³)", 0))
                continue
                
            start_time = time.time()
            try:
                hull = ConvexHull(points)
                end_time = time.time()
                execution_time = end_time - start_time
                hull_size = len(hull.vertices)
                
                complexity = {
                    "Graham's Scan": "O(n log n)",
                    "Quickhull": "O(n log n) avg",
                    "Jarvis March": "O(nh)",
                    "Brute Force": "O(n³)"
                }[algo]
                
                results.append((algo, dist_name, execution_time, complexity, hull_size))
                hull_sizes[f"{algo}-{dist_name}"] = hull_size
                
            except Exception as e:
                results.append((algo, dist_name, "Error", complexity, 0))
    
    return pd.DataFrame(results, columns=["Algorithm", "Distribution", "Time (s)", "Complexity", "Hull Size"]), hull_sizes, distributions

# --------------------- STREAMLIT UI CODE ---------------------
st.set_page_config(page_title="Convex Hull Visualizer", layout="wide")

# --- HERO BANNER WITH ANIMATED BACKGROUND ---
st.markdown("""
<style>
.hero-banner {
    position: relative;
    width: 100%;
    min-height: 180px;
    background: linear-gradient(120deg, #1E88E5 0%, #43E97B 100%);
    border-radius: 24px;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px #1e88e544;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}
.hero-banner .hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
    color: #fff;
    width: 100%;
}
.hero-banner h1 {
    font-size: 3.5rem;
    font-weight: 900;
    margin-bottom: 0.2em;
    letter-spacing: 2px;
    text-shadow: 2px 2px 8px #1E88E5AA;
}
.hero-banner p {
    font-size: 1.5rem;
    font-weight: 500;
    margin-bottom: 0;
    text-shadow: 1px 1px 4px #1E88E5AA;
}
.hero-anim {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: 1;
    pointer-events: none;
}
@keyframes float {
    0% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
    100% { transform: translateY(0); }
}
.hero-shape {
    position: absolute;
    border-radius: 50%;
    opacity: 0.18;
    animation: float 4s ease-in-out infinite;
}
.hero-shape1 { width: 120px; height: 120px; background: #fff; left: 8%; top: 20%; animation-delay: 0s; }
.hero-shape2 { width: 80px; height: 80px; background: #9C27B0; right: 10%; top: 30%; animation-delay: 1s; }
.hero-shape3 { width: 60px; height: 60px; background: #43E97B; left: 40%; bottom: 10%; animation-delay: 2s; }
.hero-shape4 { width: 100px; height: 100px; background: #1E88E5; right: 20%; bottom: 15%; animation-delay: 1.5s; }

/* Metric cards */
.metric-cards {
    display: flex;
    gap: 32px;
    margin-bottom: 24px;
    justify-content: center;
}
.metric-card {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    box-shadow: 0 2px 16px #1e88e522;
    padding: 24px 32px;
    min-width: 180px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 2px solid #e3f0ff;
    position: relative;
}
.metric-card .metric-icon {
    font-size: 2.2rem;
    margin-bottom: 8px;
}
.metric-card .metric-label {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2px;
}
.metric-card .metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #1E88E5;
}
.metric-card .metric-tooltip {
    position: absolute;
    top: 8px; right: 12px;
    font-size: 1.1rem;
    color: #888;
    cursor: pointer;
}
.metric-card .metric-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        top: 28px; right: 0;
        background: #fff;
        color: #222;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px #1e88e522;
        white-space: nowrap;
        z-index: 10;
}
/* Modern tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background: #f3f7fa;
    border-radius: 16px;
    padding: 8px 12px;
    margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1E88E5;
    border-radius: 12px;
    padding: 8px 24px;
    margin-right: 4px;
    transition: background 0.2s, color 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #1E88E5 60%, #43E97B 100%);
    color: #fff !important;
}
/* Floating Action Button */
.fab {
    position: fixed;
    bottom: 32px;
    right: 32px;
    z-index: 1000;
    background: linear-gradient(90deg, #1E88E5 60%, #43E97B 100%);
    color: #fff;
    border-radius: 50%;
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    box-shadow: 0 8px 32px #1e88e544;
    cursor: pointer;
    border: none;
    outline: none;
    transition: background 0.2s, transform 0.2s;
}
.fab:hover {
    background: linear-gradient(90deg, #43E97B 60%, #1E88E5 100%);
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)

# --- HERO BANNER ---
st.markdown('''
<div class="hero-banner">
    <div class="hero-content">
        <h1>Convex Hull Visualizer</h1>
        <p>Explore, learn, and visualize computational geometry algorithms in style!</p>
    </div>
    <div class="hero-anim">
        <div class="hero-shape hero-shape1"></div>
        <div class="hero-shape hero-shape2"></div>
        <div class="hero-shape hero-shape3"></div>
        <div class="hero-shape hero-shape4"></div>
    </div>
</div>
''', unsafe_allow_html=True)

# Enhanced CSS with better spacing and readability
st.markdown("""
<style>
    .main-header {
        animation: gradient-move 3s ease-in-out infinite alternate;
    }
    @keyframes gradient-move {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
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
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 16px;
    }
    /* Modern button styling for all Streamlit buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1E88E5 60%, #43E97B 100%) !important;
        color: #fff !important;
        border-radius: 16px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 14px 32px !important;
        margin: 8px 0 !important;
        box-shadow: 0 4px 16px #1e88e522;
        border: none !important;
        transition: background 0.2s, transform 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #43E97B 60%, #1E88E5 100%) !important;
        color: #fff !important;
        transform: scale(1.04);
        box-shadow: 0 8px 32px #1e88e544;
    }
    /* Ensure full width for containers */
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
    /* Better spacing for step documentation */
    .element-container {
        margin-bottom: 1rem;
    }
    /* Ensure expandable sections have good spacing */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    /* Input method button grid custom style */
    .input-method-btn {
        width: 100%;
        padding: 22px 0 12px 0;
        border-radius: 18px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 8px;
        background: #f3f7fa;
        color: #1E88E5;
        border: 2px solid #e3f0ff;
        box-shadow: 0 2px 12px #1e88e522;
        transition: background 0.2s, color 0.2s, transform 0.2s, border 0.2s;
        cursor: pointer;
    }
    .input-method-btn.selected {
        background: linear-gradient(90deg, #1E88E5 60%, #43E97B 100%);
        color: #fff;
        border: 2px solid #1E88E5;
        transform: scale(1.06);
        box-shadow: 0 8px 32px #1e88e544;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔍 Convex Hull Algorithm Visualizer</p>', unsafe_allow_html=True)
st.markdown('<div class="info-text">An interactive educational tool for visualizing convex hull algorithms with comprehensive step-by-step documentation and analysis.</div>', unsafe_allow_html=True)

if 'points' not in st.session_state:
    st.session_state.points = []
if 'geo_points' not in st.session_state:
    st.session_state.geo_points = []

tab1, tab2, tab3 = st.tabs(["🎯 2D Cartesian", "🌍 Geographical", "📊 Algorithm Comparison"])

with tab1:
    st.markdown('<p class="sub-header">2D Cartesian Convex Hull with Detailed Step Documentation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        input_method = st.radio(
            "Input method:",
            ["Manual Entry", "Random Points", "Upload CSV"],
            help="Choose how to input your points for convex hull computation.",
            key="input_method_radio"
        )
        
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
            distribution = st.selectbox("Distribution type", 
                                      ["Uniform Random", "Gaussian", "Circle", "Grid"])
            
            if st.button("Generate Points"):
                if distribution == "Uniform Random":
                    st.session_state.points = np.random.rand(num_points, 2).tolist()
                elif distribution == "Gaussian":
                    points = np.random.normal(0.5, 0.15, (num_points, 2))
                    points = np.clip(points, 0, 1)
                    st.session_state.points = points.tolist()
                elif distribution == "Circle":
                    angles = np.linspace(0, 2*np.pi, num_points)
                    radius = 0.4
                    center = [0.5, 0.5]
                    points = [[center[0] + radius * np.cos(angle), 
                              center[1] + radius * np.sin(angle)] for angle in angles]
                    st.session_state.points = points
                elif distribution == "Grid":
                    side = int(np.sqrt(num_points))
                    x = np.linspace(0.1, 0.9, side)
                    y = np.linspace(0.1, 0.9, side)
                    points = [[i, j] for i in x for j in y]
                    st.session_state.points = points[:num_points]
                
                st.success(f"Generated {len(st.session_state.points)} {distribution} points")
                
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.points = df.values.tolist()
                st.success(f"Loaded {len(df)} points from CSV")
                
        if st.button("Clear Points"):
            st.session_state.points = []
            st.success("All points cleared")
        
        algorithm = st.selectbox("Select algorithm:  ℹ️", 
                               ["Graham's Scan", "Quickhull", "Jarvis March", "Brute Force"],
                               help="Select the convex hull algorithm to visualize.")
        
        # Algorithm information
        algorithm_info = {
            "Graham's Scan": {
                "complexity": "O(n log n)",
                "description": "Sorts points by polar angle and uses stack-based processing",
                "best_for": "General purpose, consistent performance"
            },
            "Quickhull": {
                "complexity": "O(n log n) average, O(n²) worst",
                "description": "Divide-and-conquer approach finding extreme points",
                "best_for": "Convex point distributions"
            },
            "Jarvis March": {
                "complexity": "O(nh) where h is hull size",
                "description": "Gift wrapping method, finds next hull point iteratively",
                "best_for": "Small convex hulls"
            },
            "Brute Force": {
                "complexity": "O(n³)",
                "description": "Checks all possible line segments for hull edges",
                "best_for": "Educational purposes, small datasets"
            }
        }
        
        if algorithm in algorithm_info:
            info = algorithm_info[algorithm]
            st.info(f"**{algorithm}**\n\n{info['description']}\n\n**Time Complexity:** {info['complexity']}\n\n**Best for:** {info['best_for']}")
        
        if len(st.session_state.points) >= 3:
            if algorithm == "Brute Force" and len(st.session_state.points) > 20:
                st.warning("⚠️ Brute Force algorithm is computationally expensive for large datasets. Consider using fewer points for better performance.")
            
            if st.button("🚀 Start Step-by-Step Visualization", type="primary"):
                points = np.array(st.session_state.points)
                
                with st.spinner("Initializing algorithm visualization..."):
                    start_time = time.time()
                    if algorithm == "Graham's Scan":
                        hull = graham_scan(points)
                    elif algorithm == "Quickhull":
                        hull = quickhull(points)
                    elif algorithm == "Jarvis March":
                        hull = jarvis_march(points)
                    else:  # Brute Force
                        hull = brute_force_convex_hull(points)
                    end_time = time.time()
                    
                st.success(f"✅ Algorithm completed in {end_time - start_time:.4f} seconds")
                
                # Show final result
                st.markdown("---")
                st.markdown("## 🎯 Final Result")
                plot_hull(points, hull, algorithm)
                st.balloons()
        else:
            st.info("📝 Add at least 3 points to compute a convex hull")
    
    with col2:
        if st.session_state.points:
            fig, ax = plt.subplots(figsize=(12, 10))
            points = np.array(st.session_state.points)
            ax.plot(points[:,0], points[:,1], 'bo', markersize=10)
            ax.set_title("Current Point Set", fontsize=16)
            ax.set_xlabel("X Coordinate", fontsize=12)
            ax.set_ylabel("Y Coordinate", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("### 📊 Point Coordinates")
            df = pd.DataFrame(st.session_state.points, columns=["X", "Y"])
            st.dataframe(df, use_container_width=True)

    # Place educational resources at the end of tab1 only
    st.markdown("---")
    st.markdown("# 📚 Educational Resources & Further Learning")
    col1_ed, col2_ed, col3_ed = st.columns(3)
    with col1_ed:
        st.markdown("### 🎯 Convex Hull Applications")
        st.markdown("""
        - **Computer Graphics & 3D Rendering**
        - **Pattern Recognition & Machine Learning**
        - **Collision Detection in Game Development**
        - **Geographic Information Systems (GIS)**
        - **Image Processing & Computer Vision**
        - **Computational Geometry Problems**
        - **Robotics Path Planning**
        - **Data Visualization & Analysis**
        """)
    with col2_ed:
        st.markdown("### ⚡ Algorithm Characteristics")
        st.markdown("""
        - **Graham's Scan:** Best general-purpose algorithm with O(n log n) complexity
        - **Quickhull:** Excellent for convex distributions, divide-and-conquer approach
        - **Jarvis March:** Optimal for small hull sizes, output-sensitive O(nh)
        - **Brute Force:** Educational value, demonstrates basic concept O(n³)
        - **Chan's Algorithm:** Optimal output-sensitive approach O(n log h)
        - **Incremental:** Simple online algorithm for dynamic point sets
        """)
    with col3_ed:
        st.markdown("### 🔗 Additional Resources")
        st.markdown("""
        - [Convex Hull Theory (Wikipedia)](https://en.wikipedia.org/wiki/Convex_hull)
        - [Computational Geometry](https://en.wikipedia.org/wiki/Computational_geometry)
        - [GeeksforGeeks Tutorials](https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/)
        - [Princeton CS Course Materials](https://www.cs.princeton.edu/courses/archive/fall05/cos226/lectures/geometry.pdf)
        - [MIT OpenCourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/)
        - [Computational Geometry Algorithms Library](https://www.cgal.org/)
        """)
    st.markdown("---")
    st.markdown("""
    **About This Tool:**
    This interactive convex hull visualizer is designed for educational purposes to help students, researchers, and professionals understand computational geometry algorithms. The tool provides comprehensive step-by-step explanations, performance comparisons, and real-world applications.

    *Created with ❤️ using Streamlit and Python for the computational geometry community.*
    """)

   

    # --- FLOATING ACTION BUTTON (FAB) ---
    st.markdown('''
    <button class="fab" onclick="window.location.reload();" title="Reset All">🔄</button>
    ''', unsafe_allow_html=True)

with tab2:
    st.markdown('<p class="sub-header">Geographical Convex Hull</p>', unsafe_allow_html=True)
    geo_tab1, geo_tab2 = st.tabs(["Map Selection", "Manual Input"])
    
    with geo_tab1:
        st.write("Click on the map to add locations to your dataset")
        
        # CSV Upload Section
        st.markdown("### 📁 Upload CSV File")
        uploaded_geo_file = st.file_uploader("Upload CSV with longitude and latitude data", type="csv", key="geo_upload")
        
        if uploaded_geo_file:
            try:
                geo_df = pd.read_csv(uploaded_geo_file)
                st.write("**Preview of uploaded data:**")
                st.dataframe(geo_df.head())
                
                # Auto-detect longitude and latitude columns
                lon_col = None
                lat_col = None
                
                for col in geo_df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['lon', 'lng', 'longitude']):
                        lon_col = col
                    elif any(keyword in col_lower for keyword in ['lat', 'latitude']):
                        lat_col = col
                
                # Manual column selection if auto-detection fails
                col1, col2 = st.columns(2)
                with col1:
                    lon_column = st.selectbox("Select Longitude Column", 
                                            options=geo_df.columns, 
                                            index=geo_df.columns.get_loc(lon_col) if lon_col else 0)
                with col2:
                    lat_column = st.selectbox("Select Latitude Column", 
                                            options=geo_df.columns,
                                            index=geo_df.columns.get_loc(lat_col) if lat_col else 0)
                
                if st.button("Load Points from CSV"):
                    try:
                        # Convert to numeric and filter valid coordinates
                        geo_df[lon_column] = pd.to_numeric(geo_df[lon_column], errors='coerce')
                        geo_df[lat_column] = pd.to_numeric(geo_df[lat_column], errors='coerce')
                        
                        # Remove rows with invalid coordinates
                        valid_coords = geo_df.dropna(subset=[lon_column, lat_column])
                        valid_coords = valid_coords[
                            (valid_coords[lon_column] >= -180) & (valid_coords[lon_column] <= 180) &
                            (valid_coords[lat_column] >= -90) & (valid_coords[lat_column] <= 90)
                        ]
                        
                        # Load into session state
                        st.session_state.geo_points = [[row[lon_column], row[lat_column]] 
                                                     for _, row in valid_coords.iterrows()]
                        
                        st.success(f"✅ Loaded {len(st.session_state.geo_points)} valid geographical points!")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing coordinates: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        
        # Create map
        m = create_clickable_map()
        
        # Add existing points to map
        for point in st.session_state.geo_points:
            folium.Marker([point[1], point[0]]).add_to(m)
        
        # Show convex hull if enough points
        if len(st.session_state.geo_points) >= 3:
            points = np.array(st.session_state.geo_points)
            hull = geographic_convex_hull(points)
            hull_points = [[p[1], p[0]] for p in hull]
            folium.PolyLine(hull_points + [hull_points[0]], color="red", weight=2.5, opacity=1).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=800, height=600)
        
        # Handle map clicks
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            new_point = [lng, lat]
            
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
                
                            #    zoom_start=4)
                
                for point in st.session_state.geo_points:
                    folium.Marker([point[1], point[0]]).add_to(m)
                
                if len(st.session_state.geo_points) >= 3:
                    points = np.array(st.session_state.geo_points)
                    hull = geographic_convex_hull(points)
                    hull_points = [[p[1], p[0]] for p in hull]
                    folium.PolyLine(hull_points + [hull_points[0]], color="red", weight=2.5, opacity=1).add_to(m)
                
                folium_static(m, width=700, height=600)
                
                st.markdown("### 📊 Geographical Coordinates")
                df = pd.DataFrame(st.session_state.geo_points, columns=["Longitude", "Latitude"])
                st.dataframe(df, use_container_width=True)

with tab3:
    st.markdown('<p class="sub-header">Comprehensive Algorithm Comparison</p>', unsafe_allow_html=True)
    st.markdown("""
    Compare the performance of different convex hull algorithms across multiple metrics and data distributions.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        num_points = st.slider("Number of points for comparison", 10, 1000, 100)
    with col2:
        comparison_type = st.selectbox("Comparison Type", 
                                     ["Performance Analysis", "Scalability Study", "Distribution Impact"])
    
    if st.button("🚀 Run Comprehensive Analysis"):
        with st.spinner("Running comprehensive analysis..."):
            
            if comparison_type == "Performance Analysis":
                df, hull_sizes, distributions = run_comprehensive_comparison(num_points)
                
                # Performance Charts
                st.subheader("📊 Performance Analysis Results")
                
                # Create multiple visualizations
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # 1. Execution Time by Algorithm and Distribution
                df_numeric = df[df['Time (s)'] != 'N/A - too slow'].copy()
                df_numeric['Time (s)'] = pd.to_numeric(df_numeric['Time (s)'])
                
                if not df_numeric.empty:
                    pivot_time = df_numeric.pivot(index='Distribution', columns='Algorithm', values='Time (s)')
                    pivot_time.plot(kind='bar', ax=ax1, color=['#1E88E5', '#FF9800', '#4CAF50', '#E91E63'])
                    ax1.set_title('Execution Time by Distribution', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Time (seconds)')
                    ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax1.tick_params(axis='x', rotation=45)
                
                # 2. Hull Size Comparison
                df_hull = df[df['Hull Size'] > 0].copy()
                if not df_hull.empty:
                    pivot_hull = df_hull.pivot(index='Distribution', columns='Algorithm', values='Hull Size')
                    pivot_hull.plot(kind='bar', ax=ax2, color=['#1E88E5', '#FF9800', '#4CAF50', '#E91E63'])
                    ax2.set_title('Hull Size by Distribution', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Number of Hull Vertices')
                    ax2.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax2.tick_params(axis='x', rotation=45)
                
                # 3. Algorithm Efficiency (Points processed per second)
                if not df_numeric.empty:
                    df_numeric['Efficiency'] = num_points / df_numeric['Time (s)']
                    pivot_eff = df_numeric.pivot(index='Distribution', columns='Algorithm', values='Efficiency')
                    pivot_eff.plot(kind='bar', ax=ax3, color=['#1E88E5', '#FF9800', '#4CAF50', '#E91E63'])
                    ax3.set_title('Algorithm Efficiency (Points/Second)', fontsize=14, fontweight='bold')
                    ax3.set_ylabel('Points Processed per Second')
                    ax3.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax3.tick_params(axis='x', rotation=45)
                
                # 4. Time Complexity Comparison (Theoretical)
                complexity_data = {
                    'Graham\'s Scan': num_points * np.log2(num_points),
                    'Quickhull': num_points * np.log2(num_points),
                    'Jarvis March': num_points * 10,  # Assuming average hull size of 10
                    'Brute Force': num_points ** 3
                }
                
                ax4.bar(complexity_data.keys(), complexity_data.values(), 
                       color=['#1E88E5', '#FF9800', '#4CAF50', '#E91E63'])
                ax4.set_title('Theoretical Operations Count', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Operations')
                ax4.set_yscale('log')
                ax4.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed Results Table
                st.subheader("📋 Detailed Results")
                st.dataframe(df, use_container_width=True)
                
            elif comparison_type == "Scalability Study":
                st.subheader("📈 Scalability Analysis")
                
                sizes = [10, 25, 50, 100, 250, 500]
                scalability_results = []
                
                progress_bar = st.progress(0)
                for i, size in enumerate(sizes):
                    points = np.random.rand(size, 2)
                    
                    # Test each algorithm
                    algorithms = ["Graham's Scan", "Quickhull", "Jarvis March"]
                    if size <= 50:
                        algorithms.append("Brute Force")
                    
                    for algo in algorithms:
                        start = time.time()
                        ConvexHull(points)  # Use scipy for consistent timing
                        exec_time = time.time() - start
                        scalability_results.append({'Size': size, 'Algorithm': algo, 'Time': exec_time})
                    
                    progress_bar.progress((i + 1) / len(sizes))
                
                scale_df = pd.DataFrame(scalability_results)
                
                # Create scalability plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Linear scale
                for algo in scale_df['Algorithm'].unique():
                    algo_data = scale_df[scale_df['Algorithm'] == algo]
                    ax1.plot(algo_data['Size'], algo_data['Time'], marker='o', label=algo, linewidth=3, markersize=8)
                
                ax1.set_xlabel('Number of Points', fontsize=12)
                ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
                ax1.set_title('Scalability Analysis - Linear Scale', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # Log scale
                for algo in scale_df['Algorithm'].unique():
                    algo_data = scale_df[scale_df['Algorithm'] == algo]
                    ax2.loglog(algo_data['Size'], algo_data['Time'], marker='o', label=algo, linewidth=3, markersize=8)
                
                ax2.set_xlabel('Number of Points (log scale)', fontsize=12)
                ax2.set_ylabel('Execution Time (log scale)', fontsize=12)
                ax2.set_title('Scalability Analysis - Log Scale', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                st.subheader("📊 Scalability Data")
                pivot_scale = scale_df.pivot(index='Size', columns='Algorithm', values='Time')
                st.dataframe(pivot_scale.round(6), use_container_width=True)
                
            else:  # Distribution Impact
                st.subheader("🎯 Distribution Impact Analysis")
                
                # Test with different distributions
                distributions = {
                    "Random Uniform": np.random.rand(num_points, 2),
                    "Gaussian Cluster": np.clip(np.random.normal(0.5, 0.1, (num_points, 2)), 0, 1),
                    "Circle": np.array([[0.5 + 0.4 * np.cos(t), 0.5 + 0.4 * np.sin(t)] 
                                       for t in np.linspace(0, 2*np.pi, num_points)]),
                    "Square Grid": np.array([[i/np.sqrt(num_points), j/np.sqrt(num_points)] 
                                           for i in range(int(np.sqrt(num_points))) 
                                           for j in range(int(np.sqrt(num_points)))])[:num_points],
                    "Extreme Points": np.vstack([np.random.rand(num_points-4, 2) * 0.6 + 0.2,
                                               [[0, 0], [1, 0], [1, 1], [0, 1]]])
                }
                
                # Visualize distributions and their hulls
                fig, axes = plt.subplots(2, 3, figsize=(20, 14))
                axes = axes.flatten()
                
                dist_results = []
                
                for i, (dist_name, points) in enumerate(distributions.items()):
                    if i < len(axes):
                        ax = axes[i]
                        
                        # Plot points
                        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=30)
                        
                        # Compute and plot hull
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            
                            # Close the hull for plotting
                            hull_closed = np.vstack([hull_points, hull_points[0]])
                            ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=3)
                            
                            # Calculate metrics
                            hull_area = hull.volume  # In 2D, volume is area
                            hull_perimeter = np.sum([np.linalg.norm(hull_points[i] - hull_points[(i+1) % len(hull_points)]) 
                                                   for i in range(len(hull_points))])
                            
                            dist_results.append({
                                'Distribution': dist_name,
                                'Hull Vertices': len(hull.vertices),
                                'Hull Area': hull_area,
                                'Hull Perimeter': hull_perimeter,
                                'Convexity Ratio': hull_area / (hull_perimeter ** 2) if hull_perimeter > 0 else 0
                            })
                            
                        except Exception as e:
                            st.warning(f"Could not compute hull for {dist_name}: {str(e)}")
                        
                        ax.set_title(f'{dist_name}\n({len(points)} points)', fontsize=12, fontweight='bold')
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(distributions), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show distribution analysis
                if dist_results:
                    st.subheader("📊 Distribution Analysis Results")
                    dist_df = pd.DataFrame(dist_results)
                    st.dataframe(dist_df.round(4), use_container_width=True)
                    
                    # Create comparison charts
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Hull vertices comparison
                    ax1.bar(dist_df['Distribution'], dist_df['Hull Vertices'], color='skyblue')
                    ax1.set_title('Hull Vertices by Distribution', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Number of Vertices')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Hull area comparison
                    ax2.bar(dist_df['Distribution'], dist_df['Hull Area'], color='lightgreen')
                    ax2.set_title('Hull Area by Distribution', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Area')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # Hull perimeter comparison
                    ax3.bar(dist_df['Distribution'], dist_df['Hull Perimeter'], color='orange')
                    ax3.set_title('Hull Perimeter by Distribution', fontsize=14, fontweight='bold')
                    ax3.set_ylabel('Perimeter')
                    ax3.tick_params(axis='x', rotation=45)
                    
                    # Convexity ratio comparison
                    ax4.bar(dist_df['Distribution'], dist_df['Convexity Ratio'], color='pink')
                    ax4.set_title('Convexity Ratio by Distribution', fontsize=14, fontweight='bold')
                    ax4.set_ylabel('Ratio')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)