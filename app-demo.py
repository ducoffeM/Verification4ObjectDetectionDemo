import pandas as pd # type: ignore
import streamlit as st # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore
import sys
from PIL import Image # type: ignore

sys.path.append(".")

from utils.backend import load_gif

from utils.backend_iou import (compute_box_corners,
                               compute_extension,
                               run_and_draw)

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "About": about,
        "IoU": iou_page,
        }
    page = st.sidebar.selectbox("Tabs:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h2>A collaboration between</h2>',
            unsafe_allow_html=True,
        )
        #st.image("logos/all_logos.png")
        st.image("logos/AIRBUS_RGB.png",width=300)
        st.image("logos/logo-DEEL-BD.png",width=300)
        st.image("logos/aniti_logo.png",width=300)
        st.image("logos/onera_logo.png",width=300)

    
    

def about():

    st.title("Object Detection Model Verification")
    st.markdown(
    """

    In the realm of modern aviation, ensuring precise and safe landing procedures is paramount.
    One emerging technology that significantly enhances landing accuracy and safety is visual-based object detection.
    This tutorial delves into the application of object detection algorithms within an aeronautical context,
    specifically focusing on visual-based landing systems.
    """)

    load_gif(gif_path="img/landing_sequence.gif", streamlit=st, width=80)

    st.markdown(
        """
        --------
        """
    )
    st.image("img/problematic-verif-detection-object.png")

    st.markdown(
    """
    Object detection is a crucial aspect of computer vision that enables machines to identify and locate objects within an image or video frame.
    In aviation, this capability can be leveraged to detect and track runways notably.
    By employing advanced machine learning models and computer vision techniques,
    aircraft can autonomously interpret visual data, thereby improving landing precision
    and reducing the reliance on traditional navigation systems.
    
    **However**, trustworthy AI for visual-based landing is essential for safety and reliability.
    This involves developing models that perform accurately under various conditions, provide interpretable outputs,
    and adhere to strict regulatory standards.
    By ensuring robustness, uncertainty quantification, and maintaining interpretability,
    we can create dependable AI-driven solutions that enhance the safety and efficiency of landing procedures.

    --------
    """
    )
    
    
    #VIDEO_URL = "video/video-A350_709-2_ubd2_video-ground_truth_boxes-fps15.0.mp4"#https://www.youtube.com/watch?v=pxtzVf2Ky-I"
    #video_file = open(VIDEO_URL, 'rb')
    #video_bytes = video_file.read()

    #st.video(video_bytes)
   
    st.markdown(
    """
    What you can do our demo:

    * Verify the Intersection Over Union (IoU), aka boxes' accuracy

    """
    )

def iou_page():
    st.title("Intersection Over Union (IoU)")
    tab_names = [
    "Introduction",
    "Demo",
    ]
    (
        tab_init,
        tab_demo,
    ) = st.tabs(tab_names)

    
    with tab_init: 
        st.markdown(
        """
        
        **Basic concepts of object detection models**

        * A bounding box is a rectangle encapsulating an object of interest. In the present app, the coordinates of a bounding box correspond to the bottom left coordinate (z0, z1) and the top right coordinate (z2, z3).

        * The ground truth bounding box corresponds to the true location of an object on an image (i.e., the reference label).

        * A predicted bounding box corresponds to the location of an object predicted by a detection model.
    
        -------
        """)
        
        st.markdown(
        """
        **What is Intersection Over Union?**

        Intersection over Union (IoU) is a performance metric commonly adopted in object detection task to evaluate the accuracy of a predicted bounding box relative to the ground truth bounding box. 
        It is calculated as the ratio of the area of overlap between the predicted and ground truth boxes to the area of their union.
        """
        )
        st.image("img/iou.png")

        st.markdown(
        """
        **Neural Networks Can Be Fragile to Perturbations!**

        In the context of object detection, we focus on the model's ability to maintain accurate localization, 
        even when faced with plausible domain changes. For the object detection model, 
        "correct behavior" means two things: 1) reliably identifying the location of the runway, 
        and 2) generating a precise bounding box around it. 
        However, ensuring robustness in this process is challenging due to the evaluation 
        metric commonly used in object detection: Intersection over Union (IoU). 
        IoU measures the overlap between the predicted and true object locations, 
        but its non-convex and multidimensional nature makes it difficult 
        to apply formal verification techniques to assess the model's resilience to perturbations. 
        This fragility becomes particularly critical when small changes can cause the model to lose track of the runway.
        """
        )
        st.image("img/NN_brittleness.png")

    with tab_demo:
        st.header("Useful links")
        st.markdown("""
        :pencil: [Paper (ArXiV)](https://arxiv.org/abs/2403.08788)
        """)
        st.markdown("""
        :pencil: [Source code (Git)](https://github.com/NoCohen66/Verification4ObjectDetection/blob/main/tutorial/tutorial.ipynb)
        """)
        
        st.title("Welcome to the Object Detection Verification Demo!")
        st.markdown(
        """
       In this demo, we're going to explore how object detection models can become vulnerable to small changes or "perturbations" in the input, and how we can check their accuracy even in such cases.
       In object detection, perturbations to the input can cause the model to lose track of objects, resulting in inaccurate localization.
       We will compute a sound overestimation of the possible localization of an object, accounting for potential input variations and compute the best and worst localization in term of IoU

        **Demo Overview** 
        In this demo, you will:
        1. Load an Image: Select an image containing a single object.
        2. Localize the object: draw the ground truth bounding box (the black rectangle) on the image to represent the object as the model should ideally detect it.
        3. Draw the Perturbations: Next, you’ll need to **draw the pink area**, which represents the **possible set of localizations** where the object could appear, considering different input perturbations (such as shifts or lighting changes). This pink area is an "overestimation" of where the model might localize the object, due to input perturbations.
        4. we’ll calculate what’s known as the **worst-case localization**. This is the most extreme case of error, where the object is predicted to be in the least accurate position, considering all the possible perturbations.
        View Result: See the computed IoU value displayed, which measures the overlap between the two boxes.

        
        First, you’ll see an image showing an object (for example, a runway). Your job is to draw a rectangle around the object, which represents the ground truth — the correct position of the object as it should be detected by the model.



        """
        )
        # Option de remplissage transparent ou non
        #fill_option = st.selectbox("Choisissez le type de remplissage", ["Transparent", "Non transparent"])
        #fill_option= st.sidebar.selectbox("Choisissez le type de remplissage", ["Transparent", "Non transparent"])
        init_image = Image.open("img/how-long-airport-runway-1.jpg")

        
        fill_option ="Transparent"
        # Définir la couleur de remplissage en fonction de l'option choisie
        
        if fill_option == "Transparent":
            fill_color = "rgba(255, 165, 0, 0.1)"  # Orange transparent
        else:
            fill_color = "rgba(255, 165, 0, 1.0)"  # Orange opaque
    
        # Specify canvas parameters in application
        drawing_mode = "rect"
        
        # Initialise session state 
        #if 'step' not in st.session_state:
        st.session_state.step = 1
        #stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = "#eee"#st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Test image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)


        # Page 1: Tracer un rectangle et récupérer les coordonnées
        st.title("Draw a ground truth box")
        # Button to reset the canvas
        if st.button("Reset drawing"):
            st.rerun()
        img_upload = Image.open(bg_image) if bg_image else init_image
        img_width, img_height = img_upload.size

        canvas_result = st_canvas(
            fill_color=fill_color,  # Couleur du remplissage des rectangles
            stroke_width=3,#stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=img_upload,#None,
            update_streamlit=realtime_update,
            drawing_mode="rect",
            key="canvas1",
        )
        

        

        # Récupérer les coordonnées du rectangle
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            canvas_result.json_data["objects"]=objects[-1:]
            discard_columns=[col for col in objects.columns if col not in ["left","top","width","height"]]
            dipslay_df=objects.drop(discard_columns,axis=1)
            for col in dipslay_df.select_dtypes(include=["object"]).columns:
                dipslay_df[col] = dipslay_df[col].astype("str")
            
            if not objects.empty:
                # Keep only the last rectangle
                dipslay_df = dipslay_df.iloc[[-1]]
                st.write("GT boxes coordinates")
                st.dataframe(dipslay_df)#,columns=["left","top","width","height"])      
                gt_corners=compute_box_corners(dipslay_df)  
        
        # Fonction pour avancer à l'étape suivante
        def next_step():
            st.session_state.step += 1

        # Bouton pour passer à la deuxième étape 
        if st.button("Go to step 2",on_click=next_step):
            if not len(canvas_result.json_data["objects"]):
                #st.write("Please draw a ground truth bounding box first")
                st.markdown(
                """
                ```
                Please draw a ground truth bounding box first
                ```
                """
                )
            else:
                st.session_state.page = 2

        # Page 2: Tracer deux rectangles et récupérer les coordonnées respectives
        if "page" in st.session_state and st.session_state.page == 2:
            st.title("Localization perturbation")
            # Specify the path to your local GIF file
            gif_path = "img/iou_gif.gif"

            load_gif(gif_path=gif_path, streamlit=st, width=100)
            
            #stroke_width2 = st.sidebar.slider.slider_value = 20
            #st.sidebar.slider(25,20)
            stroke_width=30
            img_upload=Image.open(bg_image) if bg_image else init_image
            img_width, img_height = img_upload.size
            canvas_result_2 = st_canvas(
                fill_color=fill_color,  # Couleur du remplissage des rectangles
                #stroke_width=stroke_width2,
                stroke_width=stroke_width,
                stroke_color="pink",
                background_color=bg_color,
                background_image=Image.open(bg_image) if bg_image else init_image,#None,
                update_streamlit=True,
                #height=img_height,
                #width=img_width,
                drawing_mode="rect",
                key="canvas2",
            )
            
            if canvas_result_2.json_data is not None:
                # Récupérer les coordonnées du rectangle
                objects = pd.json_normalize(canvas_result_2.json_data["objects"])
                if not objects.empty:
                    discard_columns=[col for col in objects.columns if col not in ["left","top","width","height"]]
                    dipslay_df=objects.drop(discard_columns,axis=1)
                    for col in dipslay_df.select_dtypes(include=["object"]).columns:
                        dipslay_df[col] = dipslay_df[col].astype("str")
                    st.write("Interval boxes coordinates")
                    st.dataframe(dipslay_df)
            
                    # Keep only the last rectangle
                    dipslay_df = dipslay_df.iloc[[-1]]
                    ext_corners=compute_box_corners(dipslay_df) 
                    ext_interval=compute_extension(ext_corners,stroke_width=stroke_width)

                    # Add a button to the Streamlit app
                    if st.button("Run computation to get worst possible IoU given your inputs"):
                        st.title("IoU Results")
                        #run_draw_min_max_boxes(ext_interval,gt_corners)
                        img=Image.open(bg_image) if bg_image else init_image
                        iou_min,iou_max=run_and_draw(ext_interval,gt_corners,image=img)
                        st.balloons()
                  


if __name__ == "__main__":
    st.set_page_config(
        page_title="Demo: Verification of Object Detection", page_icon=":pencil2:"
    )
    #st.title("Verification of Object Detection")
    st.sidebar.subheader("Menu")
    main()
