import re
import time
import logging
from typing import Optional, Dict, List
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests
)

logger = logging.getLogger(__name__)

class TranscriptFetcher:
    def __init__(self):
        # Optional: path to cookies.txt (netscape format) to access age-restricted videos
        # e.g. export YT_COOKIES=/path/to/cookies.txt
        import os
        self.cookies_path = os.getenv("YT_COOKIES") or None
        # Optional proxy, e.g. HTTPS_PROXY=http://user:pass@host:port
        self.proxies = None
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if https_proxy:
            self.proxies = {"https": https_proxy, "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy")}

    def get_video_id(self, url: str) -> Optional[str]:
        """
        Robustly extract the 11-char YouTube video ID from common URL patterns.
        Supports: watch?v=, youtu.be/, /shorts/, /live/, /embed/
        """
        try:
            # Fast path for plain IDs
            if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
                return url

            parsed = urlparse(url)

            # Standard watch URL
            if parsed.netloc.endswith("youtube.com"):
                q = parse_qs(parsed.query)
                if "v" in q and q["v"]:
                    vid = q["v"][0]
                    if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid):
                        return vid

                # /embed/VIDEOID, /shorts/VIDEOID, /live/VIDEOID
                m = re.search(r"/(embed|shorts|live)/([0-9A-Za-z_-]{11})", parsed.path)
                if m:
                    return m.group(2)

            # youtu.be/VIDEOID
            if parsed.netloc.endswith("youtu.be"):
                m = re.match(r"^/([0-9A-Za-z_-]{11})", parsed.path)
                if m:
                    return m.group(1)

            # Fallback regex (last resort)
            m = re.search(r"([0-9A-Za-z_-]{11})", url)
            return m.group(1) if m else None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None

    def _pick_best_transcript(
        self,
        video_id: str,
        preferred_langs: List[str]
    ) -> Dict:
        """
        Use list_transcripts to select in this order:
        1) Manual transcript in any preferred language
        2) Generated transcript in any preferred language
        3) Manual transcript in original language
        4) Generated transcript in original language
        5) Translated transcript to the first preferred language (if available)
        """
        listing = YouTubeTranscriptApi.list_transcripts(
            video_id,
            proxies=self.proxies,
            cookies=self.cookies_path
        )

        # Helper to try a set of languages from available transcripts
        def first_available(manual: bool, langs: List[str]):
            for lang in langs:
                try:
                    if manual:
                        t = listing.find_manually_created_transcript([lang])
                    else:
                        t = listing.find_generated_transcript([lang])
                    return t
                except Exception:
                    continue
            return None

        # 1) Manual in preferred
        t = first_available(True, preferred_langs)
        if t: return t

        # 2) Generated in preferred
        t = first_available(False, preferred_langs)
        if t: return t

        # 3) Any manual (original language)
        try:
            for tr in listing:
                if not tr.is_generated:
                    return tr
        except Exception:
            pass

        # 4) Any generated (original language)
        try:
            for tr in listing:
                if tr.is_generated:
                    return tr
        except Exception:
            pass

        # 5) Try translating to first preferred language from *any* transcript
        if preferred_langs:
            target = preferred_langs[0]
            try:
                for tr in listing:
                    if tr.is_translatable and target in tr.translation_languages:
                        return tr.translate(target)
            except Exception:
                pass

        raise NoTranscriptFound(f"No usable transcript for video {video_id}")

    def get_transcript(self, url: str, language: str = "Auto-detect") -> Optional[Dict]:
        """
        Robust transcript getter with proper fallback order and backoff.
        language: "Thai" | "English" | "Auto-detect" (UI can still pass Thai/English)
        """
        video_id = self.get_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None

        # Map UI choice to preferred languages
        if language == "Thai":
            preferred = ["th", "en"]  # prefer Thai, then English
        elif language == "English":
            preferred = ["en"]
        else:
            preferred = ["en"]  # “Auto” → be pragmatic; English is most common

        last_err = None
        for attempt in range(1, 5):  # up to 4 tries with backoff
            try:
                transcript_obj = self._pick_best_transcript(video_id, preferred)
                transcript = transcript_obj.fetch()
                full_text = " ".join(seg.get("text", "") for seg in transcript if seg.get("text"))
                logger.info(f"Got transcript for {video_id} via {transcript_obj.language_code} (generated={getattr(transcript_obj,'is_generated',False)})")
                return {
                    "text": full_text.strip(),
                    "segments": transcript,
                    "language": transcript_obj.language_code
                }
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                logger.warning(f"No transcript for {video_id}: {e}")
                return None
            except VideoUnavailable as e:
                logger.error(f"Video unavailable {video_id}: {e}")
                return None
            except TooManyRequests as e:
                last_err = e
                sleep_s = min(2 ** attempt, 10)
                logger.warning(f"Rate limited (attempt {attempt}) for {video_id}; sleeping {sleep_s}s")
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 8)
                logger.warning(f"Transient error (attempt {attempt}) for {video_id}: {e}; sleeping {sleep_s}s")
                time.sleep(sleep_s)
                continue

        logger.error(f"Failed to get transcript for {video_id} after retries: {last_err}")
        return None
