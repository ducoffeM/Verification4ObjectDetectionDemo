import numpy as np # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
from PIL import Image # type: ignore
import base64

from typing import Any

def load_gif(gif_path:str, streamlit:Any,width:int=100)->None:

    # Load the GIF file
    with open(gif_path, "rb") as gif_file:
        gif_data = gif_file.read()

    # Encode the GIF data as a base64 string   
    encoded_gif = base64.b64encode(gif_data).decode("utf-8")

    # Create an HTML element to display the GIF
    gif_html = f'''
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="data:image/gif;base64,{encoded_gif}" alt="GIF" style="width: {width}%;"></img>
    </div>
    '''

    # Display the GIF using st.markdown with unsafe_allow_html=True
    streamlit.markdown(gif_html, unsafe_allow_html=True)

def compute_corners(df_rect: pd.DataFrame):
    """
    Compute the box coordinates from the columns 'top', 'left', 'height', 'width'
    Args:
    df (pd.DataFrame): DataFrame with the columns 'top', 'left', 'height', 'width'.
    
    Returns:
    pd.DataFrame: DataFrame with the columns 'top_left', 'top_right', 'bottom_left', 'bottom_right' with the coordinates (x, y).
    """
    def get_corners(row): #coordinates (x,y)
        top_left = (row['top'],row['left'])
        top_right = (row['top'],row['left'] + row['width'])
        bottom_left = (row['top'] + row['height'],row['left'])
        bottom_right = (row['top'] + row['height'],row['left'] + row['width'])
        #bottom_left = (left, top + height)
        #top_right = (left + width, top)
        return pd.Series({
            'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right
        })
    def get_corners2(row): #coordinates (x,y)
        top_left = (row['top'],row['left'])
        top_right = (row['top'],row['left'] + row['width'])
        bottom_left = (row['top'] + row['height'],row['left'])
        bottom_right = (row['top'] + row['height'],row['left'] + row['width'])
        #bottom_left = (left, top + height)
        #top_right = (left + width, top)
        return pd.Series({
            'top': row['top'],
            'left':row['left'],
            'bottom':row['top'] + row['height'],
            'right':row['left'] + row['width']
        })
    
    corners_df = df_rect.apply(get_corners2, axis=1)
    return pd.concat([df_rect, corners_df], axis=1)



# Convert results to a more readable format
def format_results(gt_box_min, gt_box_max):
    min_box, min_iou = gt_box_min
    max_box, max_iou = gt_box_max
    
    min_box_str = f"GT box min: {min_box} with IoU value: {min_iou[0]:.2f}"
    max_box_str = f"GT box max: {max_box} with IoU value: {max_iou[0]:.2f}"
    
    return min_box_str, max_box_str



def convert_bbox_format(bbox):
    """
    Convert bbox from [x_min, y_min, x_max, y_max] to [left, top, width, height]
    
    Parameters:
    bbox (list): A list containing [x_min, y_min, x_max, y_max]
    
    Returns:
    dict: A dictionary with keys ['left', 'top', 'width', 'height']
    """
    x_min, y_min, x_max, y_max = bbox
    left = x_min
    top = y_min
    width = x_max - x_min
    height = y_max - y_min
    return {"left": left, "top": top, "width": width, "height": height}







# List of existing color names and their RGBA values
colors = {
    "red": (255, 0, 0, 1),
    "green": (0, 255, 0, 1),
    "blue": (0, 0, 255, 1),
    "yellow": (255, 255, 0, 1),
    "cyan": (0, 255, 255, 1),
    "magenta": (255, 0, 255, 1),
    "orange": (255, 165, 0, 1),
    "purple": (128, 0, 128, 1),
    "pink": (255, 192, 203, 1),
    "brown": (165, 42, 42, 1),
}




    
