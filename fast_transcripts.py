import os
import tempfile
import ffmpeg
import numpy as np

def extract_and_split_audio(audio_file_path, chunk_duration_ms=60000):  # 60 seconds chunks
    """Extract audio from video and split into chunks"""
    try:
        # Create a temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Get audio duration using ffprobe
        probe = ffmpeg.probe(audio_file_path)
        duration = float(probe['streams'][0]['duration'])
        chunk_duration_s = chunk_duration_ms / 1000  # convert to seconds
        
        # Split audio into chunks
        for i in range(0, int(duration), int(chunk_duration_s)):
            chunk_path = os.path.join(temp_dir, f'chunk_{i}.wav')
            
            # Extract chunk using ffmpeg
            stream = ffmpeg.input(audio_file_path, ss=i, t=chunk_duration_s)
            stream = ffmpeg.output(stream, chunk_path, acodec='pcm_s16le', ar=16000, ac=1)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
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
