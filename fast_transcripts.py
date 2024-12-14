import os
import tempfile
import ffmpeg
import numpy as np
from pydub import AudioSegment

def extract_and_split_audio(audio_file_path, chunk_duration_ms=60000):  # 60 seconds chunks
    """Extract audio from video and split into chunks"""
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_wav(audio_file_path)
        
        # Create a temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunk_path = os.path.join(temp_dir, f'chunk_{i//chunk_duration_ms}.wav')
            chunk.export(chunk_path, format='wav')
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    except Exception as e:
        print(f"Error in extract_and_split_audio: {str(e)}")
        return []

async def transcribe_all_chunks(client, chunks):
    """Transcribe all audio chunks"""
    transcripts = []
    for chunk_path in chunks:
        try:
            with open(chunk_path, 'rb') as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                transcripts.append(response)
        except Exception as e:
            print(f"Error transcribing chunk: {str(e)}")
            continue
        finally:
            try:
                os.unlink(chunk_path)
            except:
                pass
    
    # Clean up the temporary directory
    try:
        if chunks and os.path.exists(os.path.dirname(chunks[0])):
            os.rmdir(os.path.dirname(chunks[0]))
    except:
        pass
    
    return transcripts

def combine_transcripts(transcripts):
    """Combine all transcripts into one text"""
    return "\n".join(filter(None, transcripts))
