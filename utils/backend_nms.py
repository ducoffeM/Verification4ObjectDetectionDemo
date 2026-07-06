import pandas as pd
import numpy as np
import streamlit as st
from .backend_iou import compute_box_corners,compute_extension_bounds
from utils.backend_iou import compute_min_iou_extension,compute_max_iou_extension
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.backend_iou import plot_rect_nms


def compute_lower_upper_matrices(stroke_width):
    n=len(st.session_state.data)
    matrix_min = np.zeros((n, n))  # Initialize the matrix with zeroes
    matrix_max = np.zeros((n, n))  # Initialize the matrix with zeroes
    #gt_corners: x,y top left, bottom right
    #display_df: left,top,width,height
    #ext_corners: x,y top left, bottom right
    for i in range(n):
        entry=st.session_state.data[i]
        """
        new_entry = {
                        "left": rect_coords["left"].values[0],
                        "top": rect_coords["top"].values[0],
                        "bottom": rect_coords["bottom_right"].values[0][0],
                        "right": rect_coords["bottom_right"].values[0][1],
                        "width": rect_coords["width"].values[0],
                        "height": rect_coords["height"].values[0],
                        "min_confidence_score": lower_bound,
                        "max_confidence_score": upper_bound
                    }
        """
        
        df_entry_i=pd.DataFrame({'top':entry['top'],
                               'left':entry['left'],
                               'height':entry['height'],
                               'width':entry['width']}, index=[0])
        
        box1_corners=compute_box_corners(df_entry_i)#'top', 'left', 'height', 'width'#
        box1_corners_bounds=compute_extension_bounds(box1_corners,stroke_width=stroke_width)# corners+ stroke width drawn
        

        for j in range(n): #compute iou between same box because it can vary?to check
            if i==j:#iou(box,box)=1
                matrix_min[i, j] = 1
                matrix_max[i, j] = 1
                continue

            #import pdb
            #pdb.set_trace()
            entry=st.session_state.data[j]
            df_entry_j=pd.DataFrame({'top':entry['top'],
                               'left':entry['left'],
                               'height': entry['height'],
                               'width':entry['width']}, index=[0])
            ext_corners=compute_box_corners(df_entry_j) #corners from box drawn
            
            """To use when working on plot section 
            ext_interval=compute_extension(ext_corners,stroke_width=stroke_width)# corners+ stroke width drawn
            iou_min,iou_max=run_and_draw(ext_interval,box1_corners,draw=False)#replace here with new function
            #complete the matrice with the right value
            matrix_min[i, j] = iou_min
            matrix_max[i, j] = iou_max
            continue
            """
            #Integration of new code to use new functions
            ##call function to compute possible values of zi to find min/max iou
            box2_corners_bounds=compute_extension_bounds(ext_corners,stroke_width=stroke_width)# corners+ stroke width drawn
            
            try:
                iou_min,best_pred_box_min_iou,best_gt_box_min_iou=compute_min_iou_extension(box1_corners_bounds, box2_corners_bounds)
            except Exception as ex:
                print(f"Got exception {ex}")
                continue
            iou_max=compute_max_iou_extension(box1_corners_bounds,box2_corners_bounds)
            if isinstance(iou_max, np.ndarray):
                iou_max = iou_max.item()  # Convert to a scalar if it's a single-element array
            #complete the matrice with the right value
            matrix_min[i, j] = iou_min

            matrix_max[i, j] = iou_max
    return matrix_min,matrix_max

def plot_final_results(data,image,boxes_map=[(1,),(2,),(0,)],canvas_height=400,canvas_width=600,colors_map=None):
    N=len(boxes_map)
    ncols=2
    nrows=N//2 +N%2

    gs = gridspec.GridSpec(nrows,ncols)
   
    # Adjust matplotlib figure size based on canvas size
    fig_width = canvas_width // ncols
    fig_height = canvas_height // nrows

    # Plot Boxes
    for i in range(nrows):
        for j in range(ncols):
            boxes_to_plot=[]
            index=i*ncols+j #i * nombre_de_colonnes + j
            #st.write(f"index {index}")
            if index >=N:
                break
            tuple_map=boxes_map[index]
            for ind_box in tuple_map:
                box_=[data[ind_box][col] for col in ["left","bottom","width","height","stroke"]]
                #normalize boxes+ change reference (repere), left and width /nrows et bottom height / ncols
                boxes_to_plot.append([round(box_[0]/ncols),#left
                                      round(fig_height-box_[1]/nrows),#bottom
                                      round(box_[2]/ncols),#width
                                      round(box_[3]/nrows),#height
                                      box_[4]#colorname
                                      ])
                
            
            ax_0 = plt.subplot(gs[i, j]) #plt.subplot(2, 1, 1)  
           
            ax_0.axis("off")
            
            if not image is None:
                image = np.array(image)  # Convert PIL image to NumPy array
                ax_0.imshow(image, transform=ax_0.transAxes, extent=[0, 1, 0, 1], aspect='auto')
            for box_plot in boxes_to_plot:
                plot_rect_nms(ax_0, np.reshape(box_plot[:4], (1, 4))[0], color=box_plot[-1], alpha=1,fc=False,linewidth=4.5)
                
            
            ax_0.set_xlim(0,fig_width)#(min(x_min[0],x_min_gt)-k, max(x_max[0],x_max_gt)+k)
            ax_0.set_ylim(0,fig_height)#(min(x_min[1],y_min_gt)-k, max(x_max[3],y_max_gt)+k)
            #ax_0.set_title()
        if index>=N:
            break
            
    plt.savefig("output/nms_plot.png")
    st.image("output/nms_plot.png")

def formal_nms(confidence_min=None, 
               confidence_max=None, 
               iou_matrix_min=None, 
               iou_matrix_max=None, #output min max iou dynamic 
               threshold=0.5):

    mask = np.zeros((1, len(confidence_min)))
    boxes = np.zeros((1, len(confidence_min)))#nb branche, one hot encoding if box selected or not
    opt_boxes = np.zeros((1, len(confidence_min)))

    max_value = np.max(confidence_max)

    while mask.min()==0:

        index = np.argmax(confidence_max[None]-max_value*mask, -1)
        index_meta, index_m = np.where(confidence_max[None] - max_value*mask >= confidence_min[index, None] )
        is_done = np.where(np.min(mask, -1)==1)[0]

    
        # filter for every candidates
        iou_matrix_meta_min = np.repeat(iou_matrix_min[None], len(index_meta)+len(is_done), 0)[index_meta, index_m]
        iou_matrix_meta_max = np.repeat(iou_matrix_max[None], len(index_meta)+len(is_done), 0)[index_meta, index_m]

        mask_meta = mask[index_meta]
        boxes_meta = boxes[index_meta]
        opt_boxes_meta = boxes[index_meta]
        
        # all the selected indices should be masked
        mask_meta[np.argsort(index_meta), index_m]=1
        boxes_meta[np.argsort(index_meta), index_m]=1
        
        # discard boxes with filtering with the current selected box
        discard = np.where((iou_matrix_meta_min>=threshold) & (mask_meta==0))
        mask_meta[discard[0], discard[1]]=1

        dual_discard = np.where((iou_matrix_meta_max>=threshold) &
                                (iou_matrix_meta_min<threshold) &
                                (mask_meta==0))

        
        if len(dual_discard[0]):

            #print('before', dual_discard)
            list_index = [[i] for i in range(len(mask_meta))]
            for i, j in zip(dual_discard[0], dual_discard[1]):
                extra_i = []
                for i_ in list_index[i]:
                    # add an index to list_index[i]
                    extra_i.append(len(mask_meta))
                    tmp = np.copy(mask_meta[i_])
                    tmp[j] = 1
                    #print(i_, j, tmp)
                    mask_meta = np.concatenate([mask_meta, tmp[None]], 0)
                    boxes_meta = np.concatenate([boxes_meta, boxes_meta[i_][None]], 0)
                    opt_boxes_meta = np.concatenate([opt_boxes_meta, opt_boxes_meta[i_][None]], 0)
                    #print(mask_meta)
                list_index[i]+=extra_i
        
        if len(is_done):
            mask = np.concatenate([mask[is_done], mask_meta]) # keep previous values
            boxes = np.concatenate([boxes[is_done], boxes_meta])
            opt_boxes = np.concatenate([opt_boxes[is_done], opt_boxes_meta])
        else:
            mask = mask_meta
            boxes = boxes_meta
            opt_boxes = opt_boxes_meta#merge

        # remove the same sequences
        dist = np.concatenate([boxes, mask, opt_boxes], -1)
        _, unique_indices = np.unique(dist, axis=0, return_index=True)
        boxes = boxes[unique_indices]
        mask = mask[unique_indices]
        opt_boxes = opt_boxes[unique_indices]

        # to do: fuse sequences depending on how many NMS sets we want to enumerate
        
    dist = np.concatenate([boxes], -1)
    _, unique_indices = np.unique(dist, axis=0, return_index=True)
    boxes = boxes[unique_indices]
    opt_boxes = opt_boxes[unique_indices]
   
    return boxes, opt_boxes

def map_boxes_to_plot(boxes):
    #return format needed to plot 
    boxes_map = []

    # Iterate over each row in the boxes array
    for row in boxes:
        # Get the indices where the value is 1
        indices = np.where(row == 1)[0]
        # Convert indices to a tuple and append to boxes_map
        boxes_map.append(tuple(indices))
    return boxes_map


