def record_audio(filename="temp_audio.wav", duration=5, fs=16000):
    print(f"[녹음 시작] {duration}초간 녹음 중...")

    import sounddevice as sd
    from scipy.io.wavfile import write

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

    print(f"[녹음 완료] 저장 위치: {filename}")