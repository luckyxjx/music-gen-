"""
Natural language emotion parser
"""

import re
from typing import Dict, Tuple, Optional


class EmotionParser:
    """Parse emotion from natural language text"""
    
    EMOTION_KEYWORDS = {
        'joy': ['happy', 'joyful', 'excited', 'cheerful', 'upbeat', 'energetic', 'positive'],
        'sadness': ['sad', 'melancholic', 'depressed', 'down', 'blue', 'unhappy', 'sorrowful'],
        'anger': ['angry', 'furious', 'mad', 'aggressive', 'intense', 'fierce'],
        'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'chill', 'mellow'],
        'surprise': ['surprised', 'shocked', 'amazed', 'unexpected'],
        'fear': ['scared', 'fearful', 'anxious', 'nervous', 'worried', 'tense']
    }
    
    EMOTION_TO_INDEX = {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'calm': 3,
        'surprise': 4,
        'fear': 5
    }
    
    def parse(self, text: str) -> Dict:
        """
        Parse emotion and duration from text
        
        Args:
            text: Natural language input (e.g., "I'm happy, give me an upbeat 3-minute track")
        
        Returns:
            Dictionary with emotion, emotion_index, duration_minutes
        """
        text = text.lower()
        
        # Parse emotion
        emotion = self._parse_emotion(text)
        emotion_index = self.EMOTION_TO_INDEX.get(emotion, 0)
        
        # Parse duration
        duration = self._parse_duration(text)
        
        return {
            'emotion': emotion,
            'emotion_index': emotion_index,
            'duration_minutes': duration,
            'text': text
        }
    
    def _parse_emotion(self, text: str) -> str:
        """Extract emotion from text"""
        # Check for each emotion's keywords
        emotion_scores = {}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score, default to 'calm'
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return 'calm'
    
    def _parse_duration(self, text: str) -> float:
        """Extract duration from text"""
        # Look for patterns like "3 minutes", "2:30", "1.5 min"
        
        # Pattern: "X minutes" or "X mins" or "X min"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins)', text)
        if match:
            return float(match.group(1))
        
        # Pattern: "X:YY" (minutes:seconds)
        match = re.search(r'(\d+):(\d+)', text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes + seconds / 60.0
        
        # Pattern: just a number followed by "track" or "song"
        match = re.search(r'(\d+)(?:-|\s)?(?:minute|min)?\s*(?:track|song)', text)
        if match:
            return float(match.group(1))
        
        # Default: 2 minutes
        return 2.0


def parse_text_input(text: str) -> Dict:
    """Convenience function to parse text input"""
    parser = EmotionParser()
    return parser.parse(text)
