import speech_recognition as sr

def listen_once():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)

    try:
        return recognizer.recognize_google(audio, language="en-IN")
    except sr.UnknownValueError:
        return "Speech not understood"
    except sr.RequestError as exc:
        return f"Speech service error: {exc}"

if __name__ == "__main__":
    print(listen_once())
