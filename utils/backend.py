from PIL import Image
import numpy as np # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
import base64
import random
import re

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

def display_images(image_paths:list,key_button,title="",option=1):
    # Load images
    images = [Image.open(img_path) for img_path in image_paths]

    # Resize images (if necessary) to ensure they are the same size
    min_width = min(img.width for img in images)
    min_height = min(img.height for img in images)
    images = [img.resize((min_width, min_height)) for img in images]

    # Convert images to numpy arrays
    image_arrays = [np.array(img) for img in images]

    # Display images sequentially with a "Next" button
    if title:
        st.title(title)

    # Initialize index for tracking current image
    index = st.session_state.get(f'image_index_{key_button}', 0)

    # Display current image
    st.image(image_arrays[index], width='stretch')#caption=image_paths[index]
    import time
    # Button to go to the next image
    if st.button(f'Next {option}',key=f'next_button_{key_button}_{index}'):
        index = (index + 1) % len(images)  # Loop back to start if at the end
        st.session_state[f'image_index_{key_button}'] = index  # Update session state
        #time.sleep(1)
   
    if st.button(f'Before {option}',key=f'next_before_{key_button}_{index}'):
        index = max((index - 1),0) % len(images)  # Loop back to start if at the end
        st.session_state[f'image_index_{key_button}'] = index  # Update session state
        #time.sleep(1)
    
    st.markdown("--------")
   
    
    """
    # Clear session state upon request
    if st.button('Clear Session State'):
        st.session_state.pop('image_index', None)
    """

def generate_random_colorname(last_colors,alpha=.3):
    # Create a set of available colors excluding those in last_colors
    available_colors = {k: v for k, v in colors.items() if k not in last_colors}
    
    # Check if there are any available colors left
    if not available_colors:
        raise ValueError("No available colors left to choose from.")
    
    # Select a random color from the available options
    random_color_name = random.choice(list(available_colors.keys()))
    rgba = available_colors[random_color_name]
    
    # Format RGBA string
    rgba_format = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {alpha})"
    stroke_color= f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"
    return random_color_name, rgba_format,stroke_color

def find_color_name(rgba_string):
    # Use regex to extract RGBA values from the string
    match = re.match(r'rgba\((\d+), (\d+), (\d+), (\d+(?:\.\d+)?)\)', rgba_string)
    if match:
        r, g, b, a = map(float, match.groups())  # Convert to float
        rgba = (int(r), int(g), int(b), a)  # Construct the RGBA tuple
    else:
        return "Invalid RGBA format"

    # Check if the given RGBA matches any color in the dictionary
    for color_name, color_value in colors.items():
        if color_value == rgba:
            return color_name
    return "Color not found"
    




    
