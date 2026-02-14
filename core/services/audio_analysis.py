"""
Audio Analysis Service — Speech-to-Text misinformation detection
Converts audio to text using SpeechRecognition, then feeds into
the existing AI analysis pipeline.
"""

import logging
import os
import tempfile
from typing import Dict

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

logger = logging.getLogger(__name__)


class AudioAnalysisService:
    """Convert audio to text using SpeechRecognition library"""

    SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.wma', '.aac', '.webm']

    def __init__(self):
        if sr is None:
            raise ImportError("SpeechRecognition is required. Install with: pip install SpeechRecognition")

        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise sensitivity
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def convert_to_wav(self, audio_path: str) -> str:
        """Convert any supported audio format to WAV for processing"""
        ext = os.path.splitext(audio_path)[1].lower()

        if ext == '.wav':
            return audio_path

        if AudioSegment is None:
            raise ImportError(
                "pydub is required for non-WAV audio files. "
                "Install with: pip install pydub\n"
                "Also ensure ffmpeg is installed on your system."
            )

        try:
            # Load the audio file
            if ext == '.mp3':
                audio = AudioSegment.from_mp3(audio_path)
            elif ext == '.ogg':
                audio = AudioSegment.from_ogg(audio_path)
            elif ext == '.flac':
                audio = AudioSegment.from_file(audio_path, format='flac')
            elif ext in ['.m4a', '.aac']:
                audio = AudioSegment.from_file(audio_path, format='m4a')
            elif ext == '.wma':
                audio = AudioSegment.from_file(audio_path, format='wma')
            elif ext == '.webm':
                audio = AudioSegment.from_file(audio_path, format='webm')
            else:
                audio = AudioSegment.from_file(audio_path)

            # Export as WAV
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format='wav')
            return wav_path

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise ValueError(f"Could not convert audio file: {e}")

    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file to text using Google Speech Recognition.
        Returns dict with transcribed_text, confidence, and metadata.
        """
        wav_path = None
        converted = False

        try:
            # Convert to WAV if needed
            wav_path = self.convert_to_wav(audio_path)
            converted = (wav_path != audio_path)

            # Load audio file
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Get audio duration
                duration = source.DURATION

                # For long audio files, process in chunks
                if duration > 60:
                    return self._transcribe_long_audio(source, duration)

                # Record the audio
                audio_data = self.recognizer.record(source)

            # Transcribe using Google's free API
            try:
                text = self.recognizer.recognize_google(audio_data)
                return {
                    'success': True,
                    'transcribed_text': text,
                    'duration_seconds': round(duration, 2),
                    'word_count': len(text.split()) if text else 0,
                    'method': 'google_speech_recognition',
                }
            except sr.UnknownValueError:
                return {
                    'success': False,
                    'transcribed_text': '',
                    'duration_seconds': round(duration, 2),
                    'word_count': 0,
                    'error': 'Could not understand the audio. The speech may be unclear or in an unsupported language.',
                    'method': 'google_speech_recognition',
                }
            except sr.RequestError as e:
                logger.error(f"Google Speech API error: {e}")
                return {
                    'success': False,
                    'transcribed_text': '',
                    'duration_seconds': round(duration, 2),
                    'word_count': 0,
                    'error': f'Speech recognition service unavailable: {e}',
                    'method': 'google_speech_recognition',
                }

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                'success': False,
                'transcribed_text': '',
                'duration_seconds': 0,
                'word_count': 0,
                'error': str(e),
            }
        finally:
            # Clean up converted file
            if converted and wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _transcribe_long_audio(self, source: 'sr.AudioFile', duration: float) -> Dict:
        """Transcribe long audio files by processing in chunks"""
        chunk_duration = 30  # seconds per chunk
        full_text = []
        offset = 0

        while offset < duration:
            remaining = duration - offset
            current_chunk = min(chunk_duration, remaining)

            try:
                audio_data = self.recognizer.record(source, duration=current_chunk)
                try:
                    chunk_text = self.recognizer.recognize_google(audio_data)
                    full_text.append(chunk_text)
                except sr.UnknownValueError:
                    # Skip chunks that can't be understood
                    full_text.append('[inaudible]')
                except sr.RequestError:
                    full_text.append('[transcription error]')
            except Exception:
                break

            offset += current_chunk

        combined = ' '.join(full_text)
        return {
            'success': bool(combined.replace('[inaudible]', '').replace('[transcription error]', '').strip()),
            'transcribed_text': combined,
            'duration_seconds': round(duration, 2),
            'word_count': len(combined.split()) if combined else 0,
            'method': 'google_speech_recognition',
            'chunks_processed': len(full_text),
        }

    def transcribe_upload(self, uploaded_file) -> Dict:
        """
        Transcribe a Django UploadedFile object.
        Saves to temp file, transcribes, then cleans up.
        """
        temp_path = None
        try:
            # Validate file extension
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                return {
                    'success': False,
                    'transcribed_text': '',
                    'error': f'Unsupported audio format: {ext}. Supported: {", ".join(self.SUPPORTED_FORMATS)}',
                    'word_count': 0,
                }

            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name

            # Transcribe
            result = self.transcribe_upload_path(temp_path)
            result['filename'] = uploaded_file.name
            result['file_size'] = uploaded_file.size
            return result

        except Exception as e:
            logger.error(f"Error processing uploaded audio: {e}")
            return {
                'success': False,
                'transcribed_text': '',
                'error': str(e),
                'word_count': 0,
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def transcribe_upload_path(self, file_path: str) -> Dict:
        """Transcribe from a file path (used internally)"""
        return self.transcribe_audio(file_path)
