import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping
    min_len = min(len(source_pts), len(target_pts))
    if min_len == 0:
        return warped_image
        
    src_pts = source_pts[:min_len]
    tgt_pts = target_pts[:min_len]

    H, W = image.shape[:2]

    corners = np.array([
        [0, 0],
        [W - 1, 0],
        [0, H - 1],
        [W - 1, H - 1]
    ])
    
    src_pts = np.vstack([src_pts, corners])
    tgt_pts = np.vstack([tgt_pts, corners])

    # Convert points to complex numbers for efficient 2D Similarity MLS formulation
    # Inverse mapping: we map target points (c) back to source points (m)
    c = tgt_pts[:, 0] + 1j * tgt_pts[:, 1]
    m = src_pts[:, 0] + 1j * src_pts[:, 1]

    # Create a grid of all pixels in the output image
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    v = grid_x.ravel() + 1j * grid_y.ravel()

    M_pixels = len(v)
    chunk_size = 100000  # Process in chunks to save memory for high-res images
    map_x = np.zeros(M_pixels, dtype=np.float32)
    map_y = np.zeros(M_pixels, dtype=np.float32)

    for i in range(0, M_pixels, chunk_size):
        v_chunk = v[i:i+chunk_size, np.newaxis]  # Shape: (chunk, 1)
        c_expand = c[np.newaxis, :]              # Shape: (1, N)

        # 1. Weights: w_i = 1 / |v - c_i|^{2\alpha}
        dist_sq = np.abs(v_chunk - c_expand)**2
        w = 1.0 / (dist_sq**alpha + eps)
        w_sum = np.sum(w, axis=1, keepdims=True)

        # 2. Weighted centroids
        c_star = np.sum(w * c_expand, axis=1, keepdims=True) / w_sum
        m_star = np.sum(w * m[np.newaxis, :], axis=1, keepdims=True) / w_sum

        # 3. Centered coordinates
        c_hat = c_expand - c_star
        m_hat = m[np.newaxis, :] - m_star

        # 4. Calculate spatial deformation matrices (Complex scalar A for Similarity transformation)
        num = np.sum(w * m_hat * np.conjugate(c_hat), axis=1)
        den = np.sum(w * np.abs(c_hat)**2, axis=1)
        A = num / (den + eps)

        # 5. Inverse mapping: v_src = m_* + A * (v - c_*)
        v_src = m_star[:, 0] + A * (v_chunk[:, 0] - c_star[:, 0])

        map_x[i:i+chunk_size] = np.real(v_src)
        map_y[i:i+chunk_size] = np.imag(v_src)

    # Reshape the mappings back to image dimensions
    map_x = map_x.reshape((H, W))
    map_y = map_y.reshape((H, W))

    # Apply remapping with OpenCV
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()