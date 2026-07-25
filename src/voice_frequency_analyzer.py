import os
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from tracker_hub import log_app_usage

def analyze_voice_spectrum(human_audio_file, ai_audio_file):
    # 음성 데이터 로드 (Streamlit 업로드 객체 직접 읽기 지원)
    y_human, sr_human = librosa.load(human_audio_file, sr=None)
    y_ai, sr_ai = librosa.load(ai_audio_file, sr=None)
    
    stft_human = np.abs(librosa.stft(y_human))
    stft_ai = np.abs(librosa.stft(y_ai))
    
    db_human = librosa.amplitude_to_db(stft_human, ref=np.max)
    db_ai = librosa.amplitude_to_db(stft_ai, ref=np.max)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    img1 = librosa.display.specshow(db_human, sr=sr_human, x_axis='time', y_axis='hz', ax=axes[0])
    axes[0].set_title('Real Human Voice (Continuous Harmonics)', fontsize=12)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # 삼항 연산자 제거 후 x_axis='time'으로 깔끔하게 정돈
    img2 = librosa.display.specshow(db_ai, sr=sr_ai, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('AI Voice (High-Frequency Cut-off)', fontsize=12)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    st.title("🎙️ AI 목소리 vs 진짜 목소리 주파수 분석기")
    st.write("음성 파일을 업로드하고 분석 버튼을 누르면 주파수 스펙트럼 차이를 확인할 수 있습니다.")

    # 앱 최초 실행 시 1회만 트래킹 로그 남김 (재실행 시 중복 방지)
    if "app_logged" not in st.session_state:
        log_app_usage("voice_frequency_analyzer", "app_opened", details={"platform": "streamlit"})
        st.session_state["app_logged"] = True

    # 파일 업로드 영역 구성을 위한 컬럼 분할
    col1, col2 = st.columns(2)
    with col1:
        human_file = st.file_uploader("1. 진짜 사람 목소리 파일 선택", type=["wav", "mp3", "m4a", "flac"])
    with col2:
        ai_file = st.file_uploader("2. AI 복제 목소리 파일 선택", type=["wav", "mp3", "m4a", "flac"])

    # 분석 시작 버튼 동작 로직
    if st.button("🔍 주파수 차이 분석 시작", type="primary"):
        if human_file and ai_file:
            log_app_usage(
                "voice_frequency_analyzer", 
                "click_analyze_button", 
                details={"human_filename": human_file.name, "ai_filename": ai_file.name}
            )
            with st.spinner("주파수를 분석하는 중입니다..."):
                analyze_voice_spectrum(human_file, ai_file)
        else:
            st.warning("두 개의 음성 파일을 모두 업로드해 주세요.")