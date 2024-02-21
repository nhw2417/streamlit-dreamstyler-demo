import streamlit as st
from PIL import Image
import torch
import custom_pipelines
from inference_style_transfer import load_model

# Streamlit 앱 제목 설정
st.title('Img2Img 스타일 변환')

# 모델 및 프로세서 로딩
sd_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
embedding_path = "embedding\overman--max_step=2000--lr=2x10-4--batch8\final.bin"
placeholder_token = "<overman>"
num_stages = 6
pipeline, processor = load_model(sd_path, controlnet_path, embedding_path, placeholder_token, num_stages)

# 사용자 입력 받기
content_image_path = st.file_uploader("변환할 이미지를 업로드하세요:", type=['png', 'jpg', 'jpeg'])
prompt = st.text_input('스타일 프롬프트를 입력하세요:', '')

if content_image_path is not None and prompt != '':
    content_image = Image.open(content_image_path)
    control_image = processor(content_image, to_pil=True)
    pos_prompt = [prompt.format(f"{placeholder_token}-T{t}") for t in range(num_stages)]
    
    # 스타일 변환 실행
    output_image = pipeline(
        prompt=pos_prompt,
        num_inference_steps=30,
        image=content_image,
        control_image=control_image,
        cross_attention_kwargs={"num_stages": num_stages},
        strength=0.8,
        guidance_scale=7.5
    ).images[0]
    
    # 결과 표시
    st.image(output_image, caption='변환된 이미지')