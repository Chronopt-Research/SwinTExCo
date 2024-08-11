import gradio as gr
from src.inference import SwinTExCo
import cv2
import os
from PIL import Image
import time
import app_config as cfg
import threading


model = SwinTExCo(weights_path=cfg.ckpt_path)

stop_thread = False

def stop_process():
    global stop_thread
    stop_thread = True

def video_colorization(video_path, ref_image, progress=gr.Progress()):
    global stop_thread
    
    # Initialize video reader
    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not video_reader.isOpened():
        gr.Warning("Please upload a valid video.")
    if ref_image is None:
        gr.Warning("Please upload a valid reference image.")
    
    # Initialize reference image
    ref_image = Image.fromarray(ref_image)
    
    # Initialize video writer
    output_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).split('.')[0] + f'_colorized_{time.time_ns()}.mp4')
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for colorized_frame, _ in zip(model.predict_video(video_reader, ref_image), progress.tqdm(range(num_frames), desc="Colorizing video", unit="frames")):
        if stop_thread:
            stop_thread = False
            break
        else:
            colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(colorized_frame)
        
    video_writer.release()
    
    return output_path

def image_colorization(image, ref_image):
    if image is None:
        gr.Warning("Please upload a valid image.")
    if ref_image is None:
        gr.Warning("Please upload a valid reference image.")
    
    # Initialize image
    image = Image.fromarray(image)
    ref_image = Image.fromarray(ref_image)
    
    colorized_image = model.predict_image(image, ref_image)
    
    return colorized_image

# app = gr.Interface(
#     fn=video_colorization,
#     inputs=[gr.Video(format="mp4", sources="upload", label="Input video (grayscale)", interactive=True), 
#             gr.Image(sources="upload", label="Reference image (color)")],
#     outputs=gr.Video(label="Output video (colorized)"),
#     title=cfg.TITLE,
#     description=cfg.DESCRIPTION,
#     allow_flagging='never'
# )

with gr.Blocks() as app:
    # Title
    gr.Markdown(cfg.CONTENT)
    
    # Video tab
    with gr.Tab("📹 Video"):
        with gr.Row():
            with gr.Column(scale=1):
                input_video_comp = gr.Video(format="mp4", sources="upload", label="Input video (grayscale)", interactive=True)
                ref_image_comp = gr.Image(sources="upload", label="Reference image (color)", height=300)
                with gr.Row():
                    with gr.Column(scale=1):
                        clear_btn = gr.ClearButton(value="Clear input", variant=['secondary'])
                        clear_btn.add([input_video_comp, ref_image_comp])
                    with gr.Column(scale=1):
                        start_btn = gr.Button(value="Start!", variant=['primary'])
            with gr.Column(scale=1):
                output_video_comp = gr.Video(label="Output video (colorized)")
                with gr.Row():
                    with gr.Column(scale=1):
                        clear_output_btn = gr.ClearButton(value="Clear output", variant=['secondary'])
                        clear_output_btn.add([output_video_comp])
                    with gr.Column(scale=1):
                        stop_btn = gr.Button(value="Stop!", variant=['stop'])
    
            start_event = start_btn.click(video_colorization, inputs=[input_video_comp, ref_image_comp], outputs=[output_video_comp])
            stop_btn.click(fn=None, cancels=[start_event])
                
    # Image tab
    with gr.Tab("🖼️ Image"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image_comp = gr.Image(sources="upload", label="Input image (grayscale)", height=300)
                ref_image_comp = gr.Image(sources="upload", label="Reference image (color)", height=300)
                with gr.Row():
                    with gr.Column(scale=1):
                        clear_input_btn = gr.ClearButton(value="Clear input", variant=['secondary'])
                        clear_input_btn.add([input_image_comp, ref_image_comp])
                    with gr.Column(scale=1):
                        start_btn = gr.Button(value="Start!", variant=['primary'])
            with gr.Column(scale=1):
                output_image_comp = gr.Image(label="Output image (colorized)", height=300)
                
                with gr.Row():
                    with gr.Column():
                        clear_output_btn = gr.ClearButton(value="Clear output", variant=['secondary'])
                        clear_output_btn.add([output_image_comp])
                    with gr.Column():
                        stop_btn = gr.Button(value="Stop!", variant=['stop'])
            
            start_event = start_btn.click(image_colorization, inputs=[input_image_comp, ref_image_comp], outputs=[output_image_comp])
            stop_btn.click(fn=None, cancels=[start_event])

    gr.Markdown(cfg.APPENDIX)

app.launch(auth=('admin', 'admin'))