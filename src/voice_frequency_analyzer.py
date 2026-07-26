import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def analyze_voice_spectrum(human_audio_file, ai_audio_file):
    # [실전 노하우 1] 재실행 관리: 파일 객체 커서 초기화 (0바이트 읽기 에러 방지)
    human_audio_file.seek(0)
    ai_audio_file.seek(0)
    
    # [실전 노하우 2] 파일 디코딩 데드락 방지: 안전하게 물리적 임시 파일로 변환
    human_ext = os.path.splitext(human_audio_file.name)[1] or ".mp3"
    ai_ext = os.path.splitext(ai_audio_file.name)[1] or ".mp3"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=human_ext) as tmp_human:
        tmp_human.write(human_audio_file.getvalue())
        human_temp_path = tmp_human.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=ai_ext) as tmp_ai:
        tmp_ai.write(ai_audio_file.getvalue())
        ai_temp_path = tmp_ai.name

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    try:
        y_human, sr_human = librosa.load(human_temp_path, sr=None)
        y_ai, sr_ai = librosa.load(ai_temp_path, sr=None)
        
        stft_human = np.abs(librosa.stft(y_human))
        stft_ai = np.abs(librosa.stft(y_ai))
        
        db_human = librosa.amplitude_to_db(stft_human, ref=np.max)
        db_ai = librosa.amplitude_to_db(stft_ai, ref=np.max)
        
        img1 = librosa.display.specshow(db_human, sr=sr_human, x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title('Real Human Voice (Continuous Harmonics)', fontsize=12)
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        img2 = librosa.display.specshow(db_ai, sr=sr_ai, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('AI Voice (High-Frequency Cut-off)', fontsize=12)
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"주파수 분석 처리 중 오류 발생: {e}")
        
    finally:
        # [실전 노하우 3] 메모리 관리: 서버 누수 방지를 위한 자원 강제 해제 및 임시 파일 삭제
        plt.close(fig)
        if os.path.exists(human_temp_path):
            os.remove(human_temp_path)
        if os.path.exists(ai_temp_path):
            os.remove(ai_temp_path)


if __name__ == "__main__":
    st.title("AI 목소리 vs 진짜 목소리 주파수 분석기")
    st.write("음성 파일을 업로드하고 분석 버튼을 누르면 주파수 스펙트럼 차이를 확인할 수 있습니다.")

    col1, col2 = st.columns(2)
    with col1:
        human_file = st.file_uploader("1. 진짜 사람 목소리 파일 선택", type=["wav", "mp3", "m4a", "flac"])
    with col2:
        ai_file = st.file_uploader("2. AI 복제 목소리 파일 선택", type=["wav", "mp3", "m4a", "flac"])

    if st.button("주파수 차이 분석 시작", type="primary"):
        if human_file and ai_file:
            
            # 자바스크립트 컴포넌트로 인한 무한 멈춤을 방지하기 위해 트래커를 우회합니다.
            # log_app_usage(
            #     "voice_frequency_analyzer", 
            #     "click_analyze_button", 
            #     details={"human_filename": human_file.name, "ai_filename": ai_file.name}
            # )
            
            with st.spinner("주파수를 분석하는 중입니다..."):
                analyze_voice_spectrum(human_file, ai_file)
        else:
            st.warning("두 개의 음성 파일을 모두 업로드해 주세요.")