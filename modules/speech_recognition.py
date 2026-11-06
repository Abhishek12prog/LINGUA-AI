
import whisper

class SpeechRecognizer:

    def __init__(self, model_name="base"):

        print("Loading Whisper Model...")

        self.model = whisper.load_model(
            model_name,
            device="cpu"
        )

        print("Whisper Model Loaded Successfully!")

    def transcribe_audio(self, audio_path):

        print("\nProcessing Audio...")

        result = self.model.transcribe(
            audio_path,
            task="transcribe"
        )

        detected_language = result["language"]
        recognized_text = result["text"]

        return detected_language, recognized_text


# Test the model
if __name__ == "__main__":

    recognizer = SpeechRecognizer()

    language, text = recognizer.transcribe_audio(
        "sample_audio/test.wav"
    )

    print("\n----------------------------")
    print("LINGUA-AI RESULTS")
    print("----------------------------")

    print("Detected Language:", language)
    print("Recognized Text:", text)
