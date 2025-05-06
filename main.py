# import streamlit as st
# from PIL import Image
# import torch
# import os
# from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from utils import image_to_canny

# st.set_page_config(page_title="AI Image Generator", layout="centered")
# st.title("🎨 AI Image Generator")

# # Dropdown for choosing the mode
# mode = st.selectbox("Choose generation mode", ["Text to Image", "Image to Image"])

# # -------------------- TEXT TO IMAGE --------------------
# if mode == "Text to Image":
#     prompt = st.text_input("Enter your prompt", value="A cute anime girl in Ghibli style")
    
#     if st.button("Generate Image"):
#         with st.spinner("Generating image from text..."):
#             pipe = StableDiffusionPipeline.from_pretrained(
#                 "runwayml/stable-diffusion-v1-5",
#                 torch_dtype=torch.float32
#             ).to("cpu")

#             result = pipe(prompt, num_inference_steps=30)
#             image = result.images[0]

#             st.image(image, caption="🖼️ Generated Image", use_column_width=True)

#             with open("text_image_output.png", "wb") as f:
#                 image.save(f)

#             with open("text_image_output.png", "rb") as file:
#                 st.download_button("Download Image", file, "text_output.png", "image/png")

# # -------------------- IMAGE TO IMAGE --------------------
# elif mode == "Image to Image":
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     prompt = st.text_input("Enter style prompt", value="Anime style boy standing in forest")

#     if uploaded_file and prompt:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Original Image", use_column_width=True)

#         if st.button("Generate from Image"):
#             with st.spinner("Generating from image..."):
#                 canny_image = image_to_canny(image)

#                 controlnet = ControlNetModel.from_pretrained(
#                     "lllyasviel/sd-controlnet-canny",
#                     torch_dtype=torch.float32
#                 )
#                 pipe = StableDiffusionControlNetPipeline.from_pretrained(
#                     "runwayml/stable-diffusion-v1-5",
#                     controlnet=controlnet,
#                     torch_dtype=torch.float32
#                 )
#                 pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#                 pipe = pipe.to("cpu")

#                 result = pipe(prompt, image=canny_image, num_inference_steps=20)
#                 generated_image = result.images[0]

#                 st.image(generated_image, caption="🎨 Generated Anime Image", use_column_width=True)

#                 with open("image2image_output.png", "wb") as f:
#                     generated_image.save(f)

#                 with open("image2image_output.png", "rb") as file:
#                     st.download_button("Download Image", file, "image2image_output.png", "image/png")

# import streamlit as st
# from PIL import Image
# import torch
# import os
# from dotenv import load_dotenv
# import openai

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from utils import image_to_canny  # This should be a canny edge function returning PIL image

# # Load .env file
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")


# # Set page config
# st.set_page_config(page_title="AI Art Generator", layout="centered")
# st.title("🎨 AI Art Generator: Text ↔ Image with Style")

# # Dropdown mode selection
# mode = st.selectbox("Choose Generation Mode", ["Text to Image", "Image to Image"])

# # ---------- TEXT TO IMAGE USING OPENAI ----------
# def generate_text_to_image():
#     st.subheader("🧠 Text to Image (OpenAI DALL·E)")

#     prompt = st.text_input("Enter your prompt", value="A cute anime girl in Ghibli style")
#     style = st.selectbox(
#         "Choose an image style",
#         ["", "realistic", "cartoon", "anime", "cyberpunk", "pixel art", "watercolor", "oil painting", "sketch", "3D render"]
#     )
#     image_size = st.selectbox("Choose image size", ["256x256", "512x512", "1024x1024"])

#     if st.button("Generate Image"):
#         if prompt:
#             with st.spinner("Generating image using OpenAI..."):
#                 try:
#                     styled_prompt = f"{prompt}, in {style} style" if style else prompt
#                     response = openai.images.generate(
#                         prompt=styled_prompt,
#                         n=1,
#                         size=image_size,
#                         response_format="url"
#                     )
#                     image_url = response.data[0].url
#                     st.image(image_url, caption=f"Generated Image ({style or 'default'} style)", use_column_width=True)
#                     st.markdown(f"[🔗 Open Image URL]({image_url})")
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#         else:
#             st.warning("Please enter a prompt.")

# # ---------- IMAGE TO IMAGE USING CONTROLNET ----------
# def generate_image_to_image():
#     st.set_page_config(page_title="Image to Image using ControlNet", layout="centered")

#     st.title("🎨 Real to Anime Image Generator with ControlNet")
#     st.markdown("Upload an image and describe the style you want (e.g., 'anime cat with big eyes').")

#     # Upload image and input prompt
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
#     prompt = st.text_input("Enter your prompt", value="anime style character")

#     if uploaded_file and prompt:
#         with st.spinner("Generating image with ControlNet..."):
#             try:
#                 # Load and display uploaded image
#                 image = Image.open(uploaded_file).convert("RGB")
#                 st.subheader("📸 Uploaded Image")
#                 st.image(image, caption="Original Uploaded Image", use_column_width=True)

#                 # Convert to Canny edges
#                 canny_image = image_to_canny(image)
#                 st.image(canny_image, caption="Canny Edge Map", use_column_width=True)

#                 # Load ControlNet model
#                 controlnet = ControlNetModel.from_pretrained(
#                     "lllyasviel/sd-controlnet-canny",
#                     torch_dtype=torch.float32  # Use float32 for CPU
#                 )
#                 pipe = StableDiffusionControlNetPipeline.from_pretrained(
#                     "runwayml/stable-diffusion-v1-5",
#                     controlnet=controlnet,
#                     torch_dtype=torch.float32
#                 )
#                 pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#                 pipe = pipe.to("cpu")  # Ensure it runs on CPU

#                 # Generate the image
#                 result = pipe(prompt, image=canny_image, num_inference_steps=20)
#                 generated_image = result.images[0]

#                 # Save and display output
#                 os.makedirs("output", exist_ok=True)
#                 out_path = os.path.join("output", "generated.png")
#                 generated_image.save(out_path)

#                 # Show result
#                 st.subheader("🎨 Generated Image")
#                 st.image(generated_image, caption="Generated Image", use_column_width=True)
#                 st.success("Done!")

#                 # Download option
#                 with open(out_path, "rb") as file:
#                     st.download_button("Download Image", file, "anime_output.png", "image/png")

#             except Exception as e:
#                 st.error(f"Generation failed: {e}")

# # ---------- MAIN SWITCH ----------
# if mode == "Text to Image":
#     generate_text_to_image()
# elif mode == "Image to Image":
#     generate_image_to_image()

# this is without styles only images
 
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# import streamlit as st
# import openai
# import os
# from dotenv import load_dotenv
 
# load_dotenv()
 
# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
 
# st.title("Text to Image Generator")
 
# prompt = st.text_input("Enter your image description:")
# image_size = st.selectbox("Choose image size", ["256x256", "512x512", "1024x1024"])
# submit = st.button("Generate Image")
 
# if submit and prompt:
#     try:
#         with st.spinner("Generating image..."):
#             response = openai.images.generate(
#                 prompt=prompt,
#                 n=1,
#                 size=image_size,
#                 response_format="url"
#             )
#             image_url = response.data[0].url
#             st.image(image_url, caption="Generated Image",use_column_width=True)
#     except Exception as e:
#         st.error(f"Error: {e}")
 
 
 
 
 
# this is with styles using dropdown
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# import streamlit as st
# import openai
# import os
# import streamlit as st
# from PIL import Image
# import torch
# import os
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# # from utils import image_to_canny
# from dotenv import load_dotenv
# import cv2
# import numpy as np

# #st.set_page_config(page_title="AI Art Generator", layout="centered")
# load_dotenv()
 
# # Set your OpenAI API key
# openai.api_key = st.secrets['OPENAI_API_KEY']
 
# st.title("!Hello , Welcome to Image Generator AI")
# mode=st.selectbox('Select Mode',['','text to image','img to img'])
 
# # utils.py

# def image_to_canny(pil_img):
#     image = np.array(pil_img)
#     image = cv2.resize(image, (512, 512))
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 100, 200)
#     canny_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
#     return Image.fromarray(canny_image)


# if mode == 'text to image':
#         # Function to generate image
#     def generate_image(prompt: str, style: str, image_size: str):
#         styled_prompt = f"{prompt}, in {style} style" if style else prompt
#         response = openai.images.generate(
#             prompt=styled_prompt,
#             n=1,
#             size=image_size,
#             response_format="url"
#         )
#         return response.data[0].url
 
#     # Streamlit UI
#     st.title(" Text to Image Generator with Style")
 
#     prompt = st.text_input("Enter your image description:")
#     style = st.selectbox(
#         "Choose an image style",
#         ["", "realistic", "cartoon", "anime", "cyberpunk", "pixel art", "watercolor", "oil painting", "sketch", "3D render"]
#     )
#     image_size = st.selectbox("Choose image size", ["256x256", "512x512", "1024x1024"])
#     submit = st.button("Generate Image")
 
#     if submit and prompt:
#         try:
#             with st.spinner("Generating image..."):
#                 image_url = generate_image(prompt, style, image_size)
#                 st.image(image_url, caption=f"Generated Image ({style or 'default'} style)", use_column_width=True)
#         except Exception as e:
#             st.error(f"Error: {e}")
            
        
# elif mode == 'img to img':
#     #st.set_page_config(page_title="Image to Anime using ControlNet", layout="centered")
 
#     st.title(" Real to Anime Image Generator with ControlNet")
#     st.markdown("Upload an image and describe the style you want (e.g., 'anime cat with big eyes').")
 
#     # Upload image and input prompt
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
#     prompt = st.text_input("Enter your prompt", ) #value="anime style cat with big eyes"
#     submit_button=st.button('generate image')
#     if submit_button and uploaded_file and prompt:
#         with st.spinner("Generating..."):
 
#             # Load and display uploaded image
#             image = Image.open(uploaded_file).convert("RGB")
#             st.subheader("📸 Uploaded Image")
#             st.image(image, caption="Original Uploaded Image", use_column_width=True)
 
#             # Convert to Canny edges
#             canny_image = image_to_canny(image)
 
#             # Load ControlNet model
#             controlnet = ControlNetModel.from_pretrained(
#                 "lllyasviel/sd-controlnet-canny",
#                 torch_dtype=torch.float32  # For CPU use
#             )
#             pipe = StableDiffusionControlNetPipeline.from_pretrained(
#                 "runwayml/stable-diffusion-v1-5",
#                 controlnet=controlnet,
#                 torch_dtype=torch.float32
#             )
#             pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
 
#             # Move model to CPU
#             pipe = pipe.to("cpu")
 
#             # Generate the image
#             result = pipe(prompt, image=canny_image, num_inference_steps=10)
#             generated_image = result.images[0]
 
#             # Save and display output
#             os.makedirs("output", exist_ok=True)
#             out_path = os.path.join("output", "generated.png")
#             generated_image.save(out_path)
 
#             # Show result
#             st.subheader("🎨 Generated Image")
#             st.image(generated_image, caption="Generated Image", use_column_width=True)
#             st.success("Done!")
 
#             # Download option
#             with open(out_path, "rb") as file:
#                 st.download_button("Download Image", file, "anime_output.png", "image/png")

from langsmith.wrappers import wrap_openai
from langsmith import traceable
import streamlit as st
import openai
import os
import streamlit as st
from PIL import Image
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from utils import image_to_canny
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
#canny function for model

def image_to_canny(pil_img):
    image = np.array(pil_img)
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    canny_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_image)
 
#st.set_page_config(page_title="AI Art Generator", layout="centered")

load_dotenv()

# st.image(image="aibg.jpeg")

st.title("Welcome to image converter")

# Set your OpenAI API key

# openai.api_key = os.getenv("OPENAI_API_KEY")
 
openai.api_key = st.secrets['OPENAI_API_KEY']
mode=st.selectbox('What you want',['Select Mode','text to image','image to image'])
 
if mode == 'text to image':
        # Function to generate image
    def generate_image(prompt: str, style: str,image_size: str):
        styled_prompt = f"{prompt}, in {style} style" if style else prompt
        response = openai.images.generate(
            prompt=styled_prompt,
            n=3, #num of img
            size="256x256",
            model="dall-e-2",
            response_format="url"
        )
        return [img_data.url for img_data in response.data]
    # Streamlit UI
    st.title(" Text to Image Generator")
    prompt = st.text_input("Enter your image description:")


    # Style options

    styles = ["realistic", "cartoon", "anime", "cyberpunk", "pixel art", "watercolor", "oil painting", "sketch", "3D render"]
    style = st.selectbox("Choose an image style", [""] + styles)
 
    # Prompt templates for styles
    style_prompts = {
    "realistic": (
        "An ultra high-resolution, photo-realistic image of {}, captured with a full-frame DSLR camera "
        "using a 50mm lens. Natural lighting, cinematic composition, soft shadows, realistic textures, "
        "depth of field, and lifelike environmental details. True-to-life color balance and reflections."
    ),
 
    "cartoon": (
        "A vibrant and expressive cartoon-style illustration of {}, featuring bold outlines, exaggerated features, "
        "and a playful color palette. Clear character design, smooth shading, and a clean, flat aesthetic suitable "
        "for animated series or comic books."
    ),
        
    "anime": (
        "A high-quality anime-style artwork of {}, with dynamic posing, dramatic lighting, and expressive facial features. "
        "Includes a detailed background in the style of modern anime, vibrant colors, cel-shading, and subtle glow effects. "
        "Inspired by Studio Ghibli and Makoto Shinkai aesthetics."
    ),
 
    "cyberpunk": (
        "A futuristic cyberpunk-themed depiction of {}, set in a neon-lit urban environment. Rich in sci-fi elements like "
        "holograms, cybernetic enhancements, rainy streets, reflective surfaces, and glowing signs. A moody, high-contrast "
        "color palette with deep purples, blues, and electric neon pinks."
    ),
 
    "pixel art": (
        "A retro 8-bit pixel art rendition of {}, with carefully crafted low-resolution details. Classic video game aesthetic, "
        "blocky textures, simplified shading, and a nostalgic color palette. Pixel-perfect outlines and sprite-style design."
    ),
 
    "watercolor": (
        "A delicate and expressive watercolor painting of {}, with fluid brush strokes, soft gradients, and subtle color blending. "
        "The image should feature paper texture, muted tones, and natural imperfections, evoking a hand-painted look "
        "with an airy, artistic feel."
    ),
 
    "oil painting": (
        "A traditional oil painting of {}, in the style of Renaissance or Baroque fine art. Rich, layered brushwork, "
        "realistic shadows and highlights, and a warm, earthy color palette. Classical composition and texture detail, "
        "with visible canvas strokes and chiaroscuro lighting."
    ),
 
    "sketch": (
        "A highly detailed pencil sketch of {}, drawn on textured paper. Includes fine linework, crosshatching, and shading to "
        "emphasize depth and contrast. Monochrome tones with subtle smudging, resembling a technical or anatomical illustration."
    ),
 
    "3D render": (
        "A professional 3D render of {}, modeled with intricate geometry and high-resolution textures. Realistic lighting setup with "
        "HDRI reflections, soft shadows, and physically accurate materials. Rendered in a photorealistic engine such as Blender Cycles "
        "or Unreal Engine."
    )
    }
 
    # Submit button

    submit = st.button("Generate Image")
 
    if submit and prompt:
        try:
            with st.spinner("Generating image..."):
                # Apply style-specific prompt
                full_prompt = style_prompts.get(style, "{}").format(prompt)
                image_urls = generate_image(full_prompt, style, "256x256")
                cols = st.columns(3)
             
                for i in range(3):
                    with cols[i]:
                        st.image(image_urls[i], caption=f"Image {i+1}")

         # st.image(image_url, caption=f"Generated Image ({style or 'default'} style)", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

elif mode == 'image to image':
    st.title("Image to Image generator")
    st.markdown("Upload an image and describe the style you want (e.g., 'anime cat with big eyes').")

    # Upload image and input prompt

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    prompt = st.text_input("Enter your prompt", ) #value="anime style cat with big eyes"
    submit_button=st.button('generate image')

    if submit_button and uploaded_file and prompt:
        with st.spinner("Generating..."):
            # Load and display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            uploaded_resized = image.resize((256, 256))
            # st.image(image, caption="Original Uploaded Image")
            # Convert to Canny edges
            canny_image = image_to_canny(image)
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float32  # For CPU use
            )

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32
            )

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            # Move model to CPU

            pipe = pipe.to("cpu")

            # Generate the image

            result = pipe(prompt, image=canny_image, num_inference_steps=5)
            generated_image = result.images[0]
            generated_resized = generated_image.resize((256, 256))
            st.subheader("📸 Uploaded vs 🎨 Generated Image")
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_resized, caption="Original Uploaded")
            with col2:
                st.image(generated_resized, caption="Generated Image")