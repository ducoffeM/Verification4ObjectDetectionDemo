
import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
from .backend import format_results,convert_bbox_format
import os 


output_path="output"
def compute_box_corners(df: pd.DataFrame):
    def get_corners(row): #coordinates (x,y)
        top_left = (row['left'],row['top'])
        top_right = (row['left'] + row['width'],row['top'])
        bottom_left = (row['left'],row['top'] + row['height'])
        bottom_right = (row['left'] + row['width'],row['top'] + row['height'])
        #bottom_left = (left, top + height)
        #top_right = (left + width, top)
        return pd.Series({
            'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right
        })
        

    corners_df = df.apply(get_corners, axis=1)
    #return x_min,y_min,x_max,y_max: bottom left and top right corners
    index=df.index[0] #if len(corners_df)<2 else -1 
    try:
        x_min = float(corners_df['top_left'][index][0])#left
        y_min = float(corners_df['top_left'][index][1])#top
        x_max = float(corners_df['bottom_right'][index][0])#right
        y_max = float(corners_df['bottom_right'][index][1])#bottom
    except KeyError as e:
        print(f"KeyError: {e} - Check if the columns 'left', 'top', 'width', 'height' exist in the DataFrame")
        raise

    
    return np.reshape(np.array([[x_min,y_min,x_max,y_max]]),(1,4))

def compute_extension(box: np.array, stroke_width:int=20):
    """
    Computes the inner box coordinates based on the outer box and stroke width.
    should return box_max et box_min for iou: [[None, 4:x_min, y_min, x_max, y_max]]*2
    return [box,box+stroke_width]
    :param outer_box: List of coordinates of the outer box [x_min, y_min, x_max, y_max]
    :param stroke_width: Width of the green stroke around the inner box
    :return: Numpy array with the corners of the outer and inner box
 
    """
   
    box_min=box+np.array([[0,0,-stroke_width,-stroke_width]])
    box_max=box+np.array([[+stroke_width,+stroke_width,0,0]])
    return [box_min,box_max]

def compute_extension_bounds(outer_box: np.array, stroke_width:int=20):
    """
    Computes the inner box coordinates based on the outer box and stroke width.

    inputs:
    box: [[None, 4:x_min, y_min, x_max, y_max]](resp. z0,z1,z2,z3)
    stroke_width: integer

    output:
    bounds_box_corner = np.array([
        [low_x0, low_y1, low_x2, low_y3],  # lower bounds of the extended box
        [up_x0, up_y1, up_x2, up_y3]   # upper bounds of the extended box
    ])   
    """
    
    def compute_inner_box(outer_box, stroke_width):
        """
        Computes the inner box coordinates based on the outer box and stroke width.
        
        :param outer_box: List of coordinates of the outer box [x_min, y_min, x_max, y_max]
        :param stroke_width: Width of the green stroke around the inner box
        :return: Numpy array with the lower and upper bounds of the box corners
        """
        x_min, y_min, x_max, y_max = outer_box[0]
    
        # Adjust coordinates for the inner box
        inner_xmin = x_min + stroke_width
        inner_ymin = y_min + stroke_width
        inner_xmax = x_max - stroke_width
        inner_ymax = y_max - stroke_width

        up_xmin = inner_xmin#x_min
        up_ymin = inner_ymin#y_min
        up_xmax = x_max
        up_ymax = y_max

        # Return the bounds in the requested format
        bounds_box_corner = np.array([
            [x_min, y_min, inner_xmax, inner_ymax],  # Lower bounds
            [up_xmin, up_ymin, up_xmax, up_ymax]       # Upper bounds
        ])
        
        return bounds_box_corner
    
    bounds_box_corners = compute_inner_box(outer_box, stroke_width)
    return bounds_box_corners

# ranges of values over the input coordinates
def iou(box, gt):
    # box_pred [None, 4:x_min, y_min, x_max, y_max] #bottom_left, top_right
    # box_gt   [None, 4:x_min^GT, y_min^GT, x_max^GT, y_max^GT] #bottom_left, top_right
    # Reshape to (N, 4) if inputs are 1-dimensional and have a length of 4
    if box.ndim == 1 and box.shape[0] == 4:
        box = box.reshape(1, 4)
    if gt.ndim == 1 and gt.shape[0] == 4:
        gt = gt.reshape(1, 4)
    
     # Check if inputs are correctly shaped
    if box.ndim != 2 or gt.ndim != 2 or box.shape[1] != 4 or gt.shape[1] != 4:
        raise ValueError("Input arrays must have shape (N, 4)")
    
    x_min = np.maximum(gt[:, 0], box[:, 0])
    y_min = np.maximum(gt[:, 1], box[:, 1])
    x_max = np.minimum(gt[:, 2], box[:, 2])
    y_max = np.minimum(gt[:, 3], box[:, 3])

    inter_area = np.maximum(0, x_max-x_min)*np.maximum(0,y_max-y_min) #check coordinate orders
    
    gt_area = (gt[:, 3]-gt[:, 1])*(gt[:, 2]-gt[:, 0])
    box_area = (box[:, 3]-box[:, 1])*(box[:, 2]-box[:, 0])

    return inter_area/np.maximum(gt_area+box_area-inter_area, 1e-4)

def argmax_iou(inputs, gt_boxes):
    # inputs= [[None, 4:x_min, y_min, x_max, y_max]]*2
    # gt_box [None, 4:_min, y_min, x_max, y_max]
    
    # compute maximum: closer to gt_boxes
    x_min = inputs[0]
    x_max = np.maximum(inputs[1], x_min)
    
    # by construction x_min <x_max
    
    
    # check gt_boxes in the range [x_min, x_max]
    dist_x_min = np.maximum(gt_boxes-x_min, 0.) # >0 if gt >x_min
    dist_x_max = np.maximum(x_max-gt_boxes, 0.) # >0 if x_max > gt
    mask_in_range = np.sign(np.minimum(dist_x_min,dist_x_max)) # >0 if x_min<gt<x_max
    mask_min = 1 - np.sign(dist_x_min) # closer to lower bound (=x_min)
    mask_max = 1 - np.sign(dist_x_max) # closer to upper bound (=x_max)
    
    sample_max = mask_in_range*gt_boxes + mask_min*x_min + (1-mask_min)*mask_max*x_max
    
    return sample_max, iou(sample_max, gt_boxes)

def get_binomial_matrix():
    matrix = np.zeros((2, 2, 2, 2, 4))
    for i_0 in range(2):
        for i_1 in range(2):
            for i_2 in range(2):
                for i_3 in range(2):
                    matrix[i_0, i_1, i_2, i_3, 0]=i_0
                    matrix[i_0, i_1, i_2, i_3, 1]=i_1
                    matrix[i_0, i_1, i_2, i_3, 2]=i_2
                    matrix[i_0, i_1, i_2, i_3, 3]=i_3
                    
    return matrix.reshape((16, 4))

def argmin_iou(inputs, gt_boxes):
    # compute maximum: closer to gt_boxes
    x_min = inputs[0] 
    x_max = np.maximum(inputs[1], x_min)

    # check if there is an overlapping in one dimension; hence iou=0
    overlapping_x = np.sign(np.maximum(x_max[:, 0]-x_min[:, 2], 0)) # >0 if x_0_max>x_1_min
    overlapping_y = np.sign(np.maximum(x_max[:, 1]-x_min[:, 3], 0)) # >0 if y_0_max > y_1_min
    
    pt_corners_x = x_min + 0.
    pt_corners_x[:, 2]= x_max[:, 0] #(=u_0)
    pt_corners_y = x_min + 0.
    pt_corners_y[:, 3]= x_max[:, 1]
        
    # compute the iou of every corners= 16 samples (others have been discarded)
    matrix = get_binomial_matrix() # (16, 4)
    corners = np.reshape(matrix[None]*x_min + (1-matrix[None])*x_max, (-1, 4))
    #gt_repeat = np.reshape(np.repeat(gt_boxes[:, None], 16, 1), (-1, 4))
    gt_repeat=np.concatenate([gt_boxes[None,:]]*16,0)
    ious = iou(corners,\
               gt_repeat)
    ious = np.reshape(ious, (-1, 16))
    index_min = np.argmin(ious, -1)
    
    # consider edge cases with overlapping
    corners = np.reshape(corners, (-1, 16, 4))
    # (None, 4)
    corner_min = corners[np.arange(len(index_min)), index_min[np.arange(len(index_min))]]
    # (None, 1)
    
    iou_min = ious[np.arange(len(index_min)), index_min[np.arange(len(index_min))]]
    mask_edge = 1 - np.maximum(overlapping_x, overlapping_y)
    return mask_edge*corner_min + overlapping_x*pt_corners_x + (1-overlapping_x)*overlapping_y*pt_corners_y,\
           mask_edge*iou_min


def get_inter(box, gt):
    # compute intersection
    x_min = np.maximum(gt[0], box[0])
    x_max = np.minimum(gt[2], box[2])
    y_min = np.maximum(gt[1], box[1])
    y_max = np.minimum(gt[3], box[3])
    return np.array([x_min, y_min, x_max, y_max])

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_rect(ax, box, color="CornflowerBlue", alpha=0.3, fc=True,linewidth=1.5):
    if fc:
        ec_ = "gray"
        fill=True
    else:
        ec_ = color
        fill = None
    rect = matplotlib.patches.Rectangle((box[0], box[1]),box[2]-box[0], box[3]-box[1],
                                        alpha = alpha, ec = ec_,
                                        fc = color, 
                                        visible = True, fill=fill,
                                        linewidth=linewidth
                                        )
    
    ax.add_patch(rect)

def plot_rect_nms(ax, box, color="CornflowerBlue", alpha=0.3, fc=True,linewidth=1.5):
    if fc:
        ec_ = "gray"
        fill=True
    else:
        ec_ = color
        fill = None
    rect = matplotlib.patches.Rectangle((box[0], box[1]),box[2], box[3],
                                        alpha = alpha, ec = ec_,
                                        fc = color, 
                                        visible = True, fill=fill,
                                        linewidth=linewidth
                                       )
    
    ax.add_patch(rect)

def run_and_draw(ext_interval,gt,draw=True,image=None,canvas_height=400,canvas_width=600):
    gt = gt[0]
    x_min_gt = np.array(gt[0].astype(float))[None]
    x_max_gt = np.array(gt[2].astype(float))[None]
    y_min_gt=np.array(gt[1].astype(float))[None]
    y_max_gt=np.array(gt[3].astype(float))[None]
    box_min,box_max = ext_interval#[box_min,box_max]
    x_0_min, y_0_min, x_1_min, y_1_min = box_min[0]
    x_0_max, y_0_max, x_1_max, y_1_max = box_max[0]
    x_min = np.maximum(np.array([x_0_min, y_0_min, x_1_min, y_1_min]), 0.)
    x_max = np.maximum(np.array([x_0_max, y_0_max, x_1_max, y_1_max]), 0.)
    
    # Check inputs to argmin_iou and argmax_iou
    box_iou_min, iou_min = argmin_iou(ext_interval, gt)
    box_iou_max, iou_max = argmax_iou(ext_interval, gt)

    # Check outputs of argmin_iou and argmax_iou
    if not draw:
        return iou_min,iou_max
    inter_min = get_inter(box_iou_min[0], np.reshape(gt, (1, 4))[0])
    inter_max = get_inter(box_iou_max[0], np.reshape(gt, (1, 4))[0])
    
    gs = gridspec.GridSpec(1,1)
    k = 0
    # Adjust matplotlib figure size based on canvas size
    dpi = 100  # Dots per inch
    fig_width = canvas_width // dpi
    fig_height = canvas_height // dpi
    #plt.figure(figsize=(fig_width, fig_height))
    
    #plt.figure(figsize=(12, 10))
    plt.axis("off")
    # Plot worst IoU
    ax_0 = plt.subplot(gs[0, 0]) #plt.subplot(2, 1, 1)  
    #ax_0.figure(figsize=(fig_width, fig_height))
    #ax_0.axis("off")
    
    if not image is None:
        #image=image.resize((fig_width,fig_height))
        image = np.array(image)  # Convert PIL image to NumPy array
        ax_0.imshow(image, transform=ax_0.transAxes, extent=[0, 1, 0, 1], aspect='auto')
        #ax_0.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]], aspect='auto')  # Display the image in the background
    plot_rect(ax_0, np.reshape(gt, (1, 4))[0], color="black", alpha=1,fc=False,linewidth=2.5)
    plot_rect(ax_0, box_iou_min[0], color="red", alpha=1,fc=False)
    #plot_rect(ax_0, inter_min, color="red", alpha=1, fc=False)
    ax_0.set_xlim(0,canvas_width)#(min(x_min[0],x_min_gt)-k, max(x_max[0],x_max_gt)+k)
    ax_0.set_ylim(0,canvas_height)#(min(x_min[1],y_min_gt)-k, max(x_max[3],y_max_gt)+k)
    ax_0.set_title("Lowest IoU ={}".format(np.round(iou_min[0], decimals=2)))
    #ax_0.legend(['GT', 'Worst Box', 'Worst Inter'], prop={'size': 6})
    
    ax_1 = ax_0#plt.subplot(gs[1, 0]) #plt.subplot(2, 1, 2)#plt.subplot(gs[0, 0])#gs[0,1]
    ax_1.axis("off")
    if not image is None:
        image = np.array(image)  # Convert PIL image to NumPy array
        ax_1.imshow(image, transform=ax_1.transAxes, extent=[0, 1, 0, 1], aspect='auto')
   
    #plot_rect(ax_1, np.reshape(gt, (1, 4))[0], color="CornflowerBlue", alpha=0.5)
    plot_rect(ax_1, box_iou_max[0], color="green", alpha=1,fc=False)
    #plot_rect(ax_1, inter_max, color="green", alpha=1, fc=False)
    #ax_1.set_xlim(0,canvas_width)#(min(x_min[0],x_min_gt)-k, max(x_max[0],x_max_gt)+k)
    #ax_1.set_ylim(0,canvas_height)#(min(x_min[1],y_min_gt)-k, max(x_max[3],y_max_gt)+k)
    ax_1.set_title("Highest IoU ={}, Lowest IoU ={}".format(np.round(iou_max[0], decimals=2),np.round(iou_min[0], decimals=2)))
    ax_1.legend(['GT', 'Worst Box', 'Best Box'], prop={'size': 6})
    area_max = (x_max[3] - x_min[1]) * (x_max[2] - x_min[0])
    # Plot worst box
    area_worst = (box_iou_min[0, 3] - box_iou_min[0, 1]) * (box_iou_min[0, 2] - box_iou_min[0, 0])
    min_box_str, max_box_str = format_results(
        (np.array(box_iou_min), np.array(iou_min)),
        (np.array([x_max]), np.array(iou_max))
    )

    converted_min_bbox = convert_bbox_format(box_iou_min[0])
    converted_max_bbox = convert_bbox_format(box_iou_max[0])
    column_order = ["left", "top", "width", "height"]

    data = {
        "Description": ["Worst Box", "Best Box"],
        "Box Coordinates (left,top,width,height)": [
            [converted_min_bbox['left'], converted_min_bbox['top'], converted_min_bbox['width'], converted_min_bbox['height']],
            [converted_max_bbox['left'], converted_max_bbox['top'], converted_max_bbox['width'], converted_max_bbox['height']]
        ],
        "IoU Value": [iou_min[0], iou_max[0]]
    }
    st.table(data)
    
    st.markdown(f"**area_max:** {area_max}, **area_worst:** {area_worst}")
    
    """
    ax_2.scatter([area_worst], [iou_min[0]], marker='^', c='red')
    # Plot best box
    area_best = (box_max[0, 3] - box_max[0, 1]) * (box_max[0, 2] - box_max[0, 0])
    ax_2.scatter([area_best], [iou_max[0]], marker='^', c='green')
    ax_2.set_xlabel('|box|')
    ax_2.set_ylabel('IoU')
    """
    plt.savefig("output/iou_min_max.png")
    st.image("output/iou_min_max.png")
    return iou_min,iou_max
   
def run_and_draw_second(ext_interval,gt,draw=True,image=None,canvas_height=400,canvas_width=600):
    gt = gt[0]
    x_min_gt = np.array(gt[0].astype(float))[None]
    x_max_gt = np.array(gt[2].astype(float))[None]
    y_min_gt=np.array(gt[1].astype(float))[None]
    y_max_gt=np.array(gt[3].astype(float))[None]
    box_max, box_min = ext_interval
    x_0_min, y_0_min, x_1_min, y_1_min = box_min[0]
    x_0_max, y_0_max, x_1_max, y_1_max = box_max[0]
    x_min = np.maximum(np.array([x_0_min, y_0_min, x_1_min, y_1_min]), 0.)
    x_max = np.maximum(np.array([x_0_max, y_0_max, x_1_max, y_1_max]), 0.)
    
    # Check inputs to argmin_iou and argmax_iou
    box_iou_min, iou_min = argmin_iou(ext_interval, gt)
    box_iou_max, iou_max = argmax_iou(ext_interval, gt)

    # Check outputs of argmin_iou and argmax_iou
    if not draw:
        return iou_min,iou_max
    inter_min = get_inter(box_iou_min[0], np.reshape(gt, (1, 4))[0])
    inter_max = get_inter(box_iou_max[0], np.reshape(gt, (1, 4))[0])
    gs = gridspec.GridSpec(1,1)
    k = 50
    # Plot best IoU
    #gs = gridspec.GridSpec(1,1)
    
    # Adjust matplotlib figure size based on canvas size
    dpi = 100  # Dots per inch
    fig_width = canvas_width / dpi
    fig_height = canvas_height / dpi
    plt.figure(figsize=(fig_width, fig_height))

    ax_1 = plt.subplot(gs[0, 0]) #plt.subplot(2, 1, 2)#plt.subplot(gs[0, 0])#gs[0,1]
    ax_1.axis("off")
    if not image is None:
        image = np.array(image)  # Convert PIL image to NumPy array
        ax_1.imshow(image, transform=ax_1.transAxes, extent=[0, 1, 0, 1], aspect='auto')
   
    plot_rect(ax_1, np.reshape(gt, (1, 4))[0], color="CornflowerBlue", alpha=0.5)
    plot_rect(ax_1, box_iou_max[0], color="green", alpha=0.1)
    plot_rect(ax_1, inter_max, color="green", alpha=1, fc=False)
    ax_1.set_xlim(min(x_min[0],x_min_gt)-k, max(x_max[0],x_max_gt)+k)
    ax_1.set_ylim(min(x_min[1],y_min_gt)-k, max(x_max[3],y_max_gt)+k)
    ax_1.set_title("Highest IoU ={}".format(np.round(iou_max[0], decimals=2)))
    ax_1.legend(['GT', 'Best Box', 'Best Inter'], prop={'size': 6})
    
    """
    # Sampling many boxes given their area
    delta_y = (iou_max[0] - iou_min[0]) / 5.0
    ax_2 = plt.subplot(gs[1, :])
    area_max = (x_max[3] - x_min[1]) * (x_max[2] - x_min[0])
    ax_2.set_xlim(-1, area_max + 100)
    ax_2.set_ylim(iou_min[0] - delta_y, iou_max[0] + delta_y)
    ax_2.plot([0, area_max], [iou_min[0], iou_min[0]], c='red')
    ax_2.plot([0, area_max], [iou_max[0], iou_max[0]], c='green')
    """
    area_max = (x_max[3] - x_min[1]) * (x_max[2] - x_min[0])
    # Plot worst box
    area_worst = (box_iou_min[0, 3] - box_iou_min[0, 1]) * (box_iou_min[0, 2] - box_iou_min[0, 0])
    min_box_str, max_box_str = format_results(
        (np.array(box_iou_min), np.array(iou_min)),
        (np.array([x_max]), np.array(iou_max))
    )

    converted_min_bbox = convert_bbox_format(box_iou_min[0])
    converted_max_bbox = convert_bbox_format(box_iou_max[0])
    column_order = ["left", "top", "width", "height"]

    data = {
        "Description": ["GT box min", "GT box max"],
        "Box Coordinates (left,top,width,height)": [
            [converted_min_bbox['left'], converted_min_bbox['top'], converted_min_bbox['width'], converted_min_bbox['height']],
            [converted_max_bbox['left'], converted_max_bbox['top'], converted_max_bbox['width'], converted_max_bbox['height']]
        ],
        "IoU Value": [iou_min[0], iou_max[0]]
    }
    st.table(data)
    
    st.markdown(f"**area_max:** {area_max}, **area_worst:** {area_worst}")
    
    """
    ax_2.scatter([area_worst], [iou_min[0]], marker='^', c='red')
    # Plot best box
    area_best = (box_max[0, 3] - box_max[0, 1]) * (box_max[0, 2] - box_max[0, 0])
    ax_2.scatter([area_best], [iou_max[0]], marker='^', c='green')
    ax_2.set_xlabel('|box|')
    ax_2.set_ylabel('IoU')
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory {output_path} created.")
    plt.savefig(f"{output_path}/iou_min_max.png")
    st.image(f"{output_path}/iou_min_max.png")
    return iou_min,iou_max


def compute_max_iou_extension(bounds_gt_corner:np.array, bounds_pred_corner:np.array):
    """
    
    Compute IoU between two dynamic boxes: Ground truth and Predicted box 
    bounds_gt_corner = np.array([
        [low_x1, low_x2, low_x3, low_x4],  # lower bounds
        [up_y1, up_y2, up_y3, up_y4]   # upper bounds
    ])
    bounds_pred_corner = np.array([
        [low_x1, low_x2, low_x3, low_x4],  # lower bounds
        [up_y1, up_y2, up_y3, up_y4]   # upper bounds
    ])

    """
    
    if bounds_gt_corner.shape != (2, 4) or bounds_pred_corner.shape != (2,4):
        raise ValueError("Matrix should be of size 2x4.")
    
    """
    top_left = (matrix[0, 0], matrix[1, 0])
    top_right = (matrix[0, 1], matrix[1, 1])
    bottom_left = (matrix[0, 2], matrix[1, 2])
    bottom_right = (matrix[0, 3], matrix[1, 3])
    """
    
    # For corner z0  (coordinates x_min,y_max of the box to use to compute MAX_IoU)
    low_gt=bounds_gt_corner[0][0]
    up_gt=bounds_gt_corner[1][0]
    low_pred=bounds_pred_corner[0][0]
    up_pred=bounds_pred_corner[1][0]

    try:
        if low_gt>=up_pred:#case 1 where : z0=x0_max,z0_gt=x0_min_gt
            z0= up_pred
            z0_gt= low_gt
        elif low_pred>=up_gt:#case 2: z0=x0_min,z0_gt=x0_max_gt
            z0= low_pred
            z0_gt=up_gt
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3.a: z0=z0gt=z0min_gt
            z0=low_gt
            z0_gt=low_gt
        elif low_pred<up_gt and up_pred>up_gt:#case 4
            #low_pred>low_gt and low_pred>up_gt:#case 4.a: z0=z0gt=z0min
            z0=low_pred
            z0_gt=low_pred
        elif low_gt> low_pred and up_gt<up_pred:#case 5: z0=z0min_gt,z0gt=z0min_gt
            z0=low_gt
            z0_gt=low_gt
        elif low_pred>low_gt and up_pred<up_gt:#case 6: z0=z0min,z0gt=z0min
            z0=low_pred
            z0_gt=low_pred
    except Exception as ex:
        print(f"Error with conditions for z0 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
    # For corner z2 (coordinates x_min,y_max of the box to use to compute MAX_IoU)
    low_gt=bounds_gt_corner[0][2]
    up_gt=bounds_gt_corner[1][2]
    low_pred=bounds_pred_corner[0][2]
    up_pred=bounds_pred_corner[1][2]
    
    try:
        if low_gt>=up_pred:#case 1 where : z2=x2_max,z2_gt=x2_min_gt
            z2= up_pred
            z2_gt= low_gt
        elif low_pred>=up_gt:#case 2: z2=x2_min,z2_gt=x2_max_gt
            z2= low_pred
            z2_gt=up_gt
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3.a: z2=z2gt=z2min_gt
            z2=low_gt
            z2_gt=low_gt
        #elif low_pred>low_gt and low_pred>up_gt:#case 4.a: z2=z2gt=z2min
        elif low_pred<up_gt and up_pred>up_gt:#case 4
            z2=low_pred
            z2_gt=low_pred
        elif low_gt> low_pred and up_gt<up_pred:#case 5: z2=z2min_gt,z2gt=z2min_gt
            z2=low_gt
            z2_gt=low_gt
        elif low_pred>low_gt and up_pred<up_gt:#case 6: z2=z2min,z2gt=z2min
            z2=low_pred
            z2_gt=low_pred
    except Exception as ex:
        print(f"Error with conditions for z2 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
    
    # For corner z1 
    low_gt=bounds_gt_corner[0][1]
    up_gt=bounds_gt_corner[1][1]
    low_pred=bounds_pred_corner[0][1]
    up_pred=bounds_pred_corner[1][1]

    try:
        if low_gt>=up_pred:#case 1
            z1=up_pred
            z1_gt=low_gt
        elif low_pred>=up_gt:#case 2
            z1= low_pred
            z1_gt=up_gt
        elif  low_gt<=up_pred and low_gt>=low_pred:#case 3.b
            z1=up_pred
            z1_gt=up_pred
        #elif low_pred>low_gt and low_pred>up_gt:#case 4.a
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a
            z1=up_gt
            z1_gt=up_gt
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            z1=up_gt
            z1_gt=up_gt
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            z1=up_pred
            z1_gt=up_pred
    except Exception as ex:
        print(f"Error with conditions for z1 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
  
    # For corner z3
    low_gt=bounds_gt_corner[0][3]
    up_gt=bounds_gt_corner[1][3]
    low_pred=bounds_pred_corner[0][3]
    up_pred=bounds_pred_corner[1][3]

    try:
        if low_gt>=up_pred:#case 1
            z3=up_pred
            z3_gt=low_gt
        elif low_pred>=up_gt:#case 2
            z3=low_pred
            z3_gt=up_gt
        elif  low_gt<=up_pred and low_gt>=low_pred:#case 3.b
            z3=up_pred
            z3_gt=up_pred
        #elif low_pred>low_gt and low_pred>up_gt:#case 4.a
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a
            z3=up_gt
            z3_gt=up_gt
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            z3=up_gt
            z3_gt=up_gt
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            z3=up_pred
            z3_gt=up_pred
    except Exception as ex:
        print(f"Error with conditions for z3 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
    #compute IoU with the right box
    box_pred=np.array([z0,z1,z2,z3])[None]#[None, 4:x_min, y_min, x_max, y_max] #bottom_left, top_right
    box_gt=np.array([z0_gt,z1_gt,z2_gt,z3_gt])[None]#[None, 4:x_min^GT, y_min^GT, x_max^GT, y_max^GT] #bottom_left, top_right
    argmax_iou=iou(box_pred,box_gt)
    return argmax_iou
   
def compute_min_iou_extension(bounds_gt_corner:np.array, bounds_pred_corner:np.array):
    """

    Compute IoU between two dynamic boxes: Ground truth and Predicted box 
    bounds_gt_corner = np.array([
        [low_z0, low_z1, low_z2, low_z3],  # lower bounds of the extended box
        [up_z0, up_z1, up_z2, up_z3]   # upper bounds of the extended box
    ])
    bounds_pred_corner = np.array([
        [low_z0, low_z1, low_z2, low_z3],  # lower bounds of the extended box
        [up_z0, up_z1, up_z2, up_z3]   # upper bounds of the extended box
    ])

    """
    
    if bounds_gt_corner.shape != (2, 4) or bounds_pred_corner.shape != (2,4):
        raise ValueError("Matrix should be of size 2x4.")
    
    zi={}
    zi_gt={}
  
    for i in range(4):
        zi[f'z{i}']=[]
        zi_gt[f'z{i}']=[]

    # For corner z0 
    low_gt=bounds_gt_corner[0][0]#first box, corner z_0
    up_gt=bounds_gt_corner[1][0]#second box, corner z_0
    low_pred=bounds_pred_corner[0][0]
    up_pred=bounds_pred_corner[1][0]

    try:
        if low_gt>=up_pred:#case 1
            zi['z0'].append(low_pred)
            zi_gt['z0'].append(up_gt)
        elif low_pred>=up_gt:#case 2
            zi['z0'].append(up_pred)
            zi_gt['z0'].append(low_gt)
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3 FAIRE CONDITION 3.a, pas de condition if else
            #3.a: gt corner in [low_gt,up_pred]
            zi['z0'].append(low_pred)
            zi_gt['z0'].append(max(up_pred,up_gt))
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a
        #elif low_pred>low_gt and low_pred>up_gt:#case 4
            zi['z0'].append(up_pred)
            zi_gt['z0'].append(max(low_pred,low_gt))
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            #case 5.a
            zi['z0'].append(low_pred)
            zi_gt['z0'].append(up_gt)
            #case 5.b 
            zi['z0'].append(up_pred)
            zi_gt['z0'].append(low_gt)
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            #case 6.a
            zi['z0'].append(low_pred)
            zi_gt['z0'].append(up_gt)
            #case 6.b
            zi['z0'].append(up_pred)
            zi_gt['z0'].append(low_gt)
    except Exception as ex:
        print(f"Error with conditions for z0 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
    ## For corner z2 
    low_gt=bounds_gt_corner[0][2]#first box, corner z_2
    up_gt=bounds_gt_corner[1][2]
    low_pred=bounds_pred_corner[0][2]
    up_pred=bounds_pred_corner[1][2]
    
    try:
        if low_gt>=up_pred:#case 1
            zi['z2'].append(low_pred)
            zi_gt['z2'].append(up_gt)
        elif low_pred>=up_gt:#case 2
            zi['z2'].append(up_pred)
            zi_gt['z2'].append(low_gt)
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3 FAIRE CONDITION 3.a, pas de condition if else
            #3.a: gt corner in [low_gt,up_pred]
            zi['z2'].append(low_pred)
            zi_gt['z2'].append(max(up_pred,up_gt))
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a   
        #elif low_pred>low_gt and low_pred>up_gt:#case 4
            zi['z2'].append(up_pred)
            zi_gt['z2'].append(max(low_pred,low_gt))
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            #case 5.a
            zi['z2'].append(low_pred)
            zi_gt['z2'].append(up_gt)
            #case 5.b 
            zi['z2'].append(up_pred)
            zi_gt['z2'].append(low_gt)
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            #case 6.a
            zi['z2'].append(low_pred)
            zi_gt['z2'].append(up_gt)
        
            #case 6.b
            zi['z2'].append(up_pred)
            zi_gt['z2'].append(low_gt)
    except Exception as ex:
        print(f"Error with conditions for z2 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
   

    # For corner z1 
    low_gt=bounds_gt_corner[0][1]#first box, corner z_1
    up_gt=bounds_gt_corner[1][1]
    low_pred=bounds_pred_corner[0][1]
    up_pred=bounds_pred_corner[1][1]

    try:
        if low_gt>=up_pred:#case 1
            zi['z1'].append(low_pred)
            zi_gt['z1'].append(up_gt)
        elif low_pred>=up_gt:#case 2
            zi['z1'].append(up_pred)
            zi_gt['z1'].append(low_gt)
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3 FAIRE CONDITION 3.a, pas de condition if else
            #3.a: gt corner in [low_gt,up_pred]
            zi['z1'].append(low_pred)
            zi_gt['z1'].append(max(up_pred,up_gt))
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a
        #elif low_pred>low_gt and low_pred>up_gt:#case 4
            zi['z1'].append(up_pred)
            zi_gt['z1'].append(max(low_pred,low_gt))
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            #case 5.a
            zi['z1'].append(low_pred)
            zi_gt['z1'].append(up_gt)
            #case 5.b 
            zi['z1'].append(up_pred)
            zi_gt['z1'].append(low_gt)
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            #case 6.a
            zi['z1'].append(low_pred)
            zi_gt['z1'].append(up_gt)
            #case 6.b
            zi['z1'].append(up_pred)
            zi_gt['z1'].append(low_gt)
    except Exception as ex:
        print(f"Error with conditions for z1 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
    # For corner z3
    low_gt=bounds_gt_corner[0][3]#first box, corner z_3
    up_gt=bounds_gt_corner[1][3]
    low_pred=bounds_pred_corner[0][3]
    up_pred=bounds_pred_corner[1][3]

    try:
        if low_gt>=up_pred:#case 1
            zi['z3'].append(low_pred)
            zi_gt['z3'].append(up_gt)
        elif low_pred>=up_gt:#case 2
            zi['z3'].append(up_pred)
            zi_gt['z3'].append(low_gt)
        elif low_gt<=up_pred and low_gt>=low_pred:#case 3 FAIRE CONDITION 3.a, pas de condition if else
            #3.a: gt corner in [low_gt,up_pred]
            zi['z3'].append(low_pred)
            zi_gt['z3'].append(max(up_pred,up_gt))
        elif low_pred<up_gt and up_pred>up_gt:#case 4.a
        #elif low_pred>low_gt and low_pred>up_gt:#case 4
            zi['z3'].append(up_pred)
            zi_gt['z3'].append(max(low_pred,low_gt))
        elif low_gt> low_pred and up_gt<up_pred:#case 5
            #case 5.a
            zi['z3'].append(low_pred)
            zi_gt['z3'].append(up_gt)
            #case 5.b 
            zi['z3'].append(up_pred)
            zi_gt['z3'].append(low_gt)
        elif low_pred>low_gt and up_pred<up_gt:#case 6
            #case 6.a
            zi['z3'].append(low_pred)
            zi_gt['z3'].append(up_gt)
            #case 6.b
            zi['z3'].append(up_pred)
            zi_gt['z3'].append(low_gt)
    except Exception as ex:
        print(f"Error with conditions for z3 low_gt={low_gt},up_gt={up_gt}, low_pred={low_pred},up_pred={up_pred}.\nException: {ex}")
   
   
    #call function to generate combination of possible boxes 
    pred_boxes, gt_boxes=generate_boxes_(zi_gt,zi)
    if pred_boxes.shape[0]==0 or gt_boxes.shape[0]==0:
        print("empty boxes")
        return 0,[],[]

    #call function to compute iou and send min/max
    try:
        ious = iou(pred_boxes, gt_boxes)
    except Exception as ex:
        print(f"Got exception iou func {ex}")
    # Find the maximum and minimum IoU values and corresponding boxes
    try:
        #min_iou = tf.reduce_min(ious)
        min_iou = np.min(ious)
    except Exception as ex:
        print(f"Got exception reduce max func {ex}")

    # Get the indices of max and min IoU
    try:
        #min_iou_idx = tf.argmin(ious)
        min_iou_idx = np.argmin(ious)
    except Exception as ex:
        print(f"Got exception argmax func {ex}")

    # Get the boxes corresponding to max/min IoU
    try:
        #best_pred_box_min_iou = pred_boxes[tf.squeeze(min_iou_idx)]
        best_pred_box_min_iou = pred_boxes[np.squeeze(min_iou_idx)]
        #best_gt_box_min_iou = gt_boxes[tf.squeeze(min_iou_idx)]
        best_gt_box_min_iou = gt_boxes[np.squeeze(min_iou_idx)]
    except Exception as ex:
        print(f"Got exception tf.squeeze func {ex}")

    # Print results
    """
    try:
        print("\nMin IoU Pred Box:", best_pred_box_min_iou)
        print("Min IoU GT Box:", best_gt_box_min_iou)
        print("Min IoU:", min_iou)
    except Exception as ex:
        print(f"Got exception : {ex}")
    """
    return min_iou,best_pred_box_min_iou,best_gt_box_min_iou




# Function to vary one corner and keep others fixed
def generate_boxes_(gt_corners,pred_corners):
    """
    i, i_gt: indices of the pred and gt corners to vary
    pred_corner_values: List of values for the i-th corner of the pred box
    gt_corner_values: List of values for the i_gt-th corner of the gt box
    """
    
    # Get the corners we are NOT varying for pred and gt
    #_pred_corners = [tf.constant(pred_corners[f'z{j}'], dtype=tf.float32) for j in range(4)] #shape (3,2)
    #_gt_corners = [tf.constant(gt_corners[f'z{j}'], dtype=tf.float32) for j in range(4)]
    _pred_corners = [pred_corners[f'z{j}'] for j in range(4)] #shape (3,2)
    _gt_corners = [gt_corners[f'z{j}'] for j in range(4)]


     
    # Create all combinations for the fixed corners
    _pred_combinations = np.stack(np.meshgrid(*_pred_corners), axis=-1)#shape meshgrid (4,2,2,2), shape stack: one array of shape [2,2,2,4]
    #operator * to split list to individual elements [a, b, c] => a,b,c
    #_gt_combinations = tf.stack(tf.meshgrid(*_gt_corners), axis=-1)
    _gt_combinations = np.stack(np.meshgrid(*_gt_corners), axis=-1)
    
    # Reshape to get them in batch format
    #_pred_combinations = tf.reshape(_pred_combinations, [-1, 4])  # 3 corners since i is varying+1, shape (8,4)
    _pred_combinations = np.reshape(_pred_combinations, [-1, 4])  # 3 corners since i is varying+1, shape (8,4)
    #_gt_combinations = tf.reshape(_gt_combinations, [-1, 4])  # 3 corners since i_gt is varying+1
    _gt_combinations = np.reshape(_gt_combinations, [-1, 4])  # 3 corners since i_gt is varying+1
    
    pred_combined = _pred_combinations#tf.concat([fixed_pred_corners, fixed_pred_combinations], axis=-1)
    gt_combined = _gt_combinations#tf.concat([fixed_gt_corners, fixed_gt_combinations], axis=-1)
    
    return pred_combined, gt_combined


