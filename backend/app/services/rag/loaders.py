import os
import json
import subprocess
import tempfile
import time
import logging as log
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from app.models.document import FileTypeEnum

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class VideoTranscriber:
    """
    Extract transcripts from videos using various methods:
    1. Google Gemini (for video understanding) - PRIMARY
    2. Whisper (OpenAI's speech-to-text model) - FALLBACK
    3. Existing subtitle files (.srt, .vtt)
    """
    
    def __init__(self, use_whisper: bool = True):
        self.use_whisper = use_whisper
        self._whisper_model = None
        self._gemini_model = None
    
    def _load_gemini(self):
        """Load Gemini model for video processing."""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                from app.core.config import settings
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self._gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                log.info("Gemini model loaded for video processing")
            except Exception as e:
                log.error(f"Error loading Gemini: {e}")
                self._gemini_model = False
        return self._gemini_model
    
    def transcribe_with_gemini(self, video_path: str) -> List[Dict[str, Any]]:
        """Transcribe video using Google Gemini."""
        model = self._load_gemini()
        if not model:
            log.warning("Gemini model not available")
            return []
        
        try:
            import google.generativeai as genai
            
            log.info(f"Uploading video to Gemini: {video_path}")
            
            # Upload the video file
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            log.info("Waiting for video processing...")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                log.error(f"Video processing failed: {video_file.state.name}")
                return []
            
            log.info("Video processed, generating transcript...")
            
            # Generate transcript with timestamps
            prompt = """Analyze this video and provide a detailed transcript of all spoken content.
            
For each segment of speech, provide:
1. The approximate start time in seconds
2. The approximate end time in seconds  
3. The spoken text

Format your response as a JSON array like this:
[
    {"start": 0, "end": 5, "text": "spoken content here"},
    {"start": 5, "end": 12, "text": "more spoken content"},
    ...
]

If there is no speech, describe what is happening in the video at each 10-second interval.
Only output the JSON array, no other text."""

            response = model.generate_content(
                [video_file, prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            try:
                segments = json.loads(response.text)
                log.info(f"Gemini transcribed {len(segments)} segments")
                
                # Clean up uploaded file
                try:
                    genai.delete_file(video_file.name)
                except:
                    pass
                
                return segments
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse Gemini response: {e}")
                log.info(f"Raw response: {response.text[:500]}")
                
                # Try to extract text content anyway
                text = response.text.strip()
                if text:
                    # Create single segment with all content
                    duration = self._get_video_duration(video_path) or 60
                    return [{
                        "start": 0,
                        "end": duration,
                        "text": text
                    }]
                return []
                
        except Exception as e:
            log.error(f"Gemini transcription error: {e}")
            return []

    def _load_whisper(self):
        """Lazy load whisper model."""
        if self._whisper_model is None and self.use_whisper:
            try:
                import whisper
                # Use base model for balance of speed/accuracy
                self._whisper_model = whisper.load_model("base")
                log.info("Whisper model loaded successfully")
            except ImportError:
                log.warning("Whisper not installed. Run: pip install openai-whisper")
                self._whisper_model = False
            except Exception as e:
                log.error(f"Error loading Whisper: {e}")
                self._whisper_model = False
        return self._whisper_model
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video using ffmpeg."""
        try:
            # Create temp file for audio
            audio_path = tempfile.mktemp(suffix=".wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(audio_path):
                log.info(f"Audio extracted to: {audio_path}")
                return audio_path
            else:
                log.error(f"FFmpeg error: {result.stderr}")
                return None
        except Exception as e:
            log.error(f"Error extracting audio: {e}")
            return None
    
    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration
        except Exception as e:
            log.error(f"Error getting video duration: {e}")
        return None
    
    def transcribe_with_whisper(self, video_path: str) -> List[Dict[str, Any]]:
        """Transcribe video using Whisper."""
        model = self._load_whisper()
        if not model:
            return []
        
        # Extract audio
        audio_path = self._extract_audio(video_path)
        if not audio_path:
            return []
        
        try:
            # Transcribe
            result = model.transcribe(audio_path, verbose=False)
            
            # Convert segments to our format
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip(),
                    'confidence': seg.get('avg_logprob', 0)
                })
            
            log.info(f"Transcribed {len(segments)} segments from video")
            return segments
        except Exception as e:
            log.error(f"Whisper transcription error: {e}")
            return []
        finally:
            # Cleanup temp audio
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
    
    def load_subtitle_file(self, subtitle_path: str) -> List[Dict[str, Any]]:
        """Load existing subtitle file (.srt or .vtt)."""
        segments = []
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT format
            if subtitle_path.endswith('.srt'):
                import re
                pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\Z)'
                matches = re.findall(pattern, content)
                
                for match in matches:
                    start = self._parse_srt_time(match[1])
                    end = self._parse_srt_time(match[2])
                    text = match[3].replace('\n', ' ').strip()
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'text': text,
                        'confidence': 1.0
                    })
            
            # Parse VTT format
            elif subtitle_path.endswith('.vtt'):
                import re
                pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n([\s\S]*?)(?=\n\n|\Z)'
                matches = re.findall(pattern, content)
                
                for match in matches:
                    start = self._parse_vtt_time(match[0])
                    end = self._parse_vtt_time(match[1])
                    text = match[2].replace('\n', ' ').strip()
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'text': text,
                        'confidence': 1.0
                    })
            
            log.info(f"Loaded {len(segments)} segments from subtitle file")
        except Exception as e:
            log.error(f"Error loading subtitle file: {e}")
        
        return segments
    
    def _parse_srt_time(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        parts = time_str.replace(',', '.').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _parse_vtt_time(self, time_str: str) -> float:
        """Convert VTT timestamp to seconds."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


class VideoLoader:
    """Load video content by extracting and chunking transcripts."""
    
    def __init__(self, file_path: str, subtitle_path: Optional[str] = None):
        self.file_path = file_path
        self.subtitle_path = subtitle_path
        self.transcriber = VideoTranscriber()
    
    def load(self) -> List[Document]:
        """Load video and return chunked transcript as Documents."""
        documents = []
        segments = []
        
        # Try to load existing subtitles first
        if self.subtitle_path and os.path.exists(self.subtitle_path):
            log.info(f"Loading subtitle file: {self.subtitle_path}")
            segments = self.transcriber.load_subtitle_file(self.subtitle_path)
        else:
            # Check for subtitle file with same name as video
            video_dir = os.path.dirname(self.file_path)
            video_name = os.path.splitext(os.path.basename(self.file_path))[0]
            
            for ext in ['.srt', '.vtt']:
                potential_sub = os.path.join(video_dir, video_name + ext)
                if os.path.exists(potential_sub):
                    log.info(f"Found subtitle file: {potential_sub}")
                    segments = self.transcriber.load_subtitle_file(potential_sub)
                    break
        
        # If no subtitles, try Gemini first (better for video understanding)
        if not segments:
            log.info("No subtitles found, trying Gemini transcription...")
            segments = self.transcriber.transcribe_with_gemini(self.file_path)
        
        # Fall back to Whisper if Gemini fails
        if not segments:
            log.info("Gemini failed, trying Whisper transcription...")
            segments = self.transcriber.transcribe_with_whisper(self.file_path)
        
        if not segments:
            log.warning(f"No transcript available for video: {self.file_path}")
            return []
        
        # Get video duration
        duration = self.transcriber._get_video_duration(self.file_path)
        
        # Group segments into chunks (roughly 30 seconds each)
        chunk_duration = 30.0  # seconds
        current_chunk = []
        current_start = 0
        
        for seg in segments:
            current_chunk.append(seg['text'])
            
            # Check if we've exceeded chunk duration
            if seg['end'] - current_start >= chunk_duration:
                chunk_text = ' '.join(current_chunk)
                
                # Calculate "page" number based on time (for consistency with PDF)
                page_num = int(current_start / chunk_duration) + 1
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'source': self.file_path,
                        'page': page_num,
                        'page_label': str(page_num),
                        'start_time': current_start,
                        'end_time': seg['end'],
                        'type': 'video_transcript',
                        'total_duration': duration
                    }
                )
                documents.append(doc)
                
                current_chunk = []
                current_start = seg['end']
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            page_num = int(current_start / chunk_duration) + 1
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    'source': self.file_path,
                    'page': page_num,
                    'page_label': str(page_num),
                    'start_time': current_start,
                    'end_time': segments[-1]['end'] if segments else current_start,
                    'type': 'video_transcript',
                    'total_duration': duration
                }
            )
            documents.append(doc)
        
        log.info(f"Loaded {len(documents)} document chunks from video")
        return documents


class DocumentLoader:
    """Unified loader for PDFs and Videos."""
    
    def __init__(self, file_path: str, file_type: FileTypeEnum, subtitle_path: Optional[str] = None):
        self.file_path = file_path
        self.file_type = file_type
        self.subtitle_path = subtitle_path

    def load(self) -> List[Document]:
        if self.file_type == FileTypeEnum.PDF:
            try:
                loader = PyPDFLoader(self.file_path)
                log.info(f"Loading PDF: {self.file_path}")
                return loader.load()
            except Exception as e:
                log.error(f"Error loading PDF file: {e}")
                return []
        
        elif self.file_type == FileTypeEnum.VIDEO:
            try:
                loader = VideoLoader(self.file_path, self.subtitle_path)
                log.info(f"Loading Video: {self.file_path}")
                return loader.load()
            except Exception as e:
                log.error(f"Error loading video file: {e}")
                return []
        
        else:
            log.error(f"Unsupported file type: {self.file_type}")
            return []