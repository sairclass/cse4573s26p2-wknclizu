'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image
# debug
from utils import write_image
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img_keys = list(imgs.keys())
    # Convert uint8 to float32 in [0, 1] and add batch dimension [B, C, H, W]
    img1 = (imgs[img_keys[0]].float() / 255.0).unsqueeze(0)
    img2 = (imgs[img_keys[1]].float() / 255.0).unsqueeze(0)

    # 1. Feature Extraction (SIFT)
    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)

    sift = K.feature.SIFTFeature(num_features=2000)
    lafs1, _, descs1 = sift(gray1)
    lafs2, _, descs2 = sift(gray2)

    # 2. Feature Matching (Ratio Test via SNN)
    distances, indexes = K.feature.match_snn(descs1[0], descs2[0], 0.8)

    # Extract XY coordinates of matches
    pts1 = K.feature.get_laf_center(lafs1)[0, indexes[:, 0]] # [N, 2]
    pts2 = K.feature.get_laf_center(lafs2)[0, indexes[:, 1]] # [N, 2]

    # DEBUG 1: Visualizing Extracted SIFT Points 
    print("DEBUG: Showing Image 1 with Matched Keypoints")
    img1_viz = img1[0].clone()
    h_viz, w_viz = img1_viz.shape[1], img1_viz.shape[2]
    for p in pts1.long():
        x, y = p[0].item(), p[1].item()
        if 0 <= x < w_viz and 0 <= y < h_viz:
            img1_viz[0, max(0, y-1):min(h_viz, y+2), max(0, x-1):min(w_viz, x+2)] = 1.0
            img1_viz[1, max(0, y-1):min(h_viz, y+2), max(0, x-1):min(w_viz, x+2)] = 0.0
            img1_viz[2, max(0, y-1):min(h_viz, y+2), max(0, x-1):min(w_viz, x+2)] = 0.0
    img1_viz_uint8 = (img1_viz * 255.0).clamp(0, 255).to(torch.uint8)
    write_image(img1_viz_uint8, "debug1_features.png")
    # =====

    # 3. Homography Estimation with RANSAC
    max_inliers = 0
    best_H = torch.eye(3, dtype=torch.float32)
    num_matches = pts1.shape[0]
    
    for _ in range(1000):
        indices = torch.randperm(num_matches)[:4]
        src_pts = pts2[indices].unsqueeze(0)
        dst_pts = pts1[indices].unsqueeze(0)
        
        try:
            H_est = K.geometry.homography.find_homography_dlt(src_pts, dst_pts)[0]
            pts2_homo = torch.cat([pts2, torch.ones_like(pts2[:, :1])], dim=-1)
            proj = (H_est @ pts2_homo.T).T
            proj = proj[:, :2] / (proj[:, 2:] + 1e-8)
            
            error = torch.norm(proj - pts1, dim=-1)
            inliers = error < 3.0
            num_inliers = inliers.sum().item()
            
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H_est
        except:
            continue

    # 4. Canvas Calculation and Transformation
    h1, w1 = img1.shape[2], img1.shape[3]
    h2, w2 = img2.shape[2], img2.shape[3]

    corners2 = torch.tensor([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=torch.float32)
    corners2_homo = torch.cat([corners2, torch.ones_like(corners2[:, :1])], dim=-1)
    proj_corners = (best_H @ corners2_homo.T).T
    proj_corners = proj_corners[:, :2] / (proj_corners[:, 2:] + 1e-8)

    all_x = torch.cat([proj_corners[:, 0], torch.tensor([0.0, w1])])
    all_y = torch.cat([proj_corners[:, 1], torch.tensor([0.0, h1])])

    min_x, min_y = torch.floor(torch.min(all_x)).item(), torch.floor(torch.min(all_y)).item()
    max_x, max_y = torch.ceil(torch.max(all_x)).item(), torch.ceil(torch.max(all_y)).item()

    out_w, out_h = int(max_x - min_x), int(max_y - min_y)

    T = torch.tensor([[1.0, 0.0, -min_x],
                      [0.0, 1.0, -min_y],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    H_img2 = T @ best_H
    H_img1 = T

    warp2 = K.geometry.transform.warp_perspective(img2, H_img2.unsqueeze(0), dsize=(out_h, out_w))
    warp1 = K.geometry.transform.warp_perspective(img1, H_img1.unsqueeze(0), dsize=(out_h, out_w))

    # DEBUG 2: Visualizing Warped Images
    # If warp2 looks like a correctly projected perspective of img2, RANSAC and Canvas logic succeeded.
    print("DEBUG: Showing Warped Image 2")
    warp2_uint8 = (warp2[0] * 255.0).clamp(0, 255).to(torch.uint8)
    write_image(warp2_uint8, "debug2_warped.png")
    # 

    mask2 = K.geometry.transform.warp_perspective(torch.ones_like(img2), H_img2.unsqueeze(0), dsize=(out_h, out_w)) > 0
    mask1 = K.geometry.transform.warp_perspective(torch.ones_like(img1), H_img1.unsqueeze(0), dsize=(out_h, out_w)) > 0

    # 5. Foreground Elimination (CUT Strategy via Pixel Difference)
    overlap = mask1 & mask2
    
    diff = torch.abs(warp1 - warp2).mean(dim=1, keepdim=True)
    moving_object_mask = diff > 0.15 

    # DEBUG 3: Visualizing Background Elimination Logic
    print("DEBUG: Saving Difference Mask and Hard Cut Mask to disk...")
    
    # 1. Save normalized difference map
    diff_vis = (diff[0] / (diff.max() + 1e-8) * 255.0).clamp(0, 255).to(torch.uint8)
    diff_vis = diff_vis.repeat(3, 1, 1) 
    write_image(diff_vis, "debug3_diff_map.png")
    
    # 2. Save the boolean moving object mask
    # Black = Background, White = Moving Object
    mask_vis = (moving_object_mask[0].float() * 255.0).clamp(0, 255).to(torch.uint8)
    mask_vis = mask_vis.repeat(3, 1, 1)
    write_image(mask_vis, "debug4_moving_mask.png")
    # 
    
    blend = (warp1 + warp2) / 2.0
    
    out_img = torch.zeros_like(warp1)
    
    out_img = torch.where(mask1 & ~mask2, warp1, out_img)
    out_img = torch.where(mask2 & ~mask1, warp2, out_img)
    
    overlap_result = torch.where(moving_object_mask, warp1, blend)
    out_img = torch.where(overlap, overlap_result, out_img)

    final_img = (out_img[0] * 255.0).clamp(0, 255).to(torch.uint8)

    return final_img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
